//! The renderer's own full-screen passes on the 3D image (`options.rs`):
//! ambient occlusion before the game's post passes, then modern bloom and
//! FXAA before the HUD. They write colour only: the alpha
//! channel holds the GS's destination alpha, which later draws test and
//! blend with.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::OnceLock;

use crate::options::{Aa, Bloom, Options};

/// Pass uniforms (`P` in post.wgsl).
#[derive(Clone, Copy, Default)]
struct U {
    texel: [f32; 2],
    /// AO: target pixels per AO pixel across (2 at half resolution).
    stride: u32,
    strength: f32,
    dir: [f32; 2],
    size: [f32; 2],
    /// Ambient occlusion with the terrain camera: world → screen, screen →
    /// world (columns), offsets, depth parameters.
    w2s: [[f32; 4]; 4],
    s2w: [[f32; 4]; 4],
    off: [f32; 4],
    zp: [f32; 4],
    /// Sun shadows: rebased world → light clip (columns), towards the sun
    /// and the strength, (texel in UV, normal offset, -, -).
    light: [[f32; 4]; 4],
    sun: [f32; 4],
    sp: [f32; 4],
    /// Ambient occlusion with the terrain camera: the camera's rays
    /// ([`ao_rays`]).
    ray: [[f32; 4]; 4],
}

/// Uniform buffer size (`P` in post.wgsl).
const U_BYTES: usize = 352;

impl U {
    fn bytes(&self) -> Vec<u8> {
        let mut w: Vec<u32> = vec![
            self.texel[0].to_bits(),
            self.texel[1].to_bits(),
            self.stride,
            self.strength.to_bits(),
            self.dir[0].to_bits(),
            self.dir[1].to_bits(),
            self.size[0].to_bits(),
            self.size[1].to_bits(),
        ];
        for m in [&self.w2s, &self.s2w] {
            w.extend(m.iter().flatten().map(|v| v.to_bits()));
        }
        for v in [&self.off, &self.zp] {
            w.extend(v.iter().map(|v| v.to_bits()));
        }
        w.extend(self.light.iter().flatten().map(|v| v.to_bits()));
        for v in [&self.sun, &self.sp] {
            w.extend(v.iter().map(|v| v.to_bits()));
        }
        w.extend(self.ray.iter().flatten().map(|v| v.to_bits()));
        debug_assert_eq!(w.len() * 4, U_BYTES);
        w.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
}

/// How a pass writes its target.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Write {
    /// Replace RGB, keep alpha (the main target).
    Rgb,
    /// Replace everything (intermediate textures).
    All,
    /// Add (intermediate textures: the bloom upsample).
    Add,
    /// RGB multiplied by the output (the AO), alpha kept.
    Multiply,
}

/// Debug views (`SSX3_GFX_SHOW`): the shadow pass's (the first five,
/// [`debug_view`]) and the AO mask (`ao`).
const VIEWS: &[&str] = &[
    "shadow-map",
    "world",
    "normal",
    "light-uv",
    "light-depth",
    "ao",
];

/// The view shown: 1 + its index in [`VIEWS`], 0 for none. From
/// `SSX3_GFX_SHOW`, changed while running by [`set_show`].
fn view() -> &'static AtomicU8 {
    static S: OnceLock<AtomicU8> = OnceLock::new();
    S.get_or_init(|| {
        let v = ssx3_core::config::var("SSX3_GFX_SHOW").unwrap_or_default();
        AtomicU8::new(view_code(&v))
    })
}

fn view_code(v: &str) -> u8 {
    VIEWS
        .iter()
        .position(|w| *w == v.trim())
        .map_or(0, |i| i as u8 + 1)
}

/// Show debug view `v` (`SSX3_GFX_SHOW`'s values; empty: none).
pub(crate) fn set_show(v: &str) {
    view().store(view_code(v), Ordering::Relaxed);
}

/// Debug: `SSX3_GFX_SHOW=ao` shows the AO mask as the image.
pub(crate) fn show(which: &str) -> bool {
    match view().load(Ordering::Relaxed) {
        0 => false,
        c => VIEWS[c as usize - 1] == which,
    }
}

/// The shadow pass's debug views (`SSX3_GFX_SHOW=shadow-map|world|normal|
/// light-uv|light-depth`: 1..5, 0 for none).
pub(crate) fn debug_view() -> f32 {
    match view().load(Ordering::Relaxed) {
        c @ 1..=5 => c as f32,
        _ => 0.0,
    }
}

/// The cascade the debug views show (`SSX3_SHADOW_CASCADE`, the widest by
/// default).
fn shadow_cascade() -> usize {
    ssx3_core::config::var("SSX3_SHADOW_CASCADE")
        .and_then(|v| v.parse().ok())
        .unwrap_or(crate::shadow::CASCADES - 1)
        .min(crate::shadow::CASCADES - 1)
}

/// Formats of the intermediate textures.
const BLOOM_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const AO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;
/// The AO's view depth: 32-bit, as positions are rebuilt from it.
const AO_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
/// Bloom pyramid levels (half size down to 1/64).
const BLOOM_LEVELS: usize = 6;
/// Uniform buffers per frame (each pass uses its own).
const UNIFORMS: usize = 32;

pub struct Post {
    module: wgpu::ShaderModule,
    layout: wgpu::BindGroupLayout,
    pipe_layout: wgpu::PipelineLayout,
    pipes: HashMap<(&'static str, wgpu::TextureFormat, Write), wgpu::RenderPipeline>,
    samp: wgpu::Sampler,
    ubufs: Vec<wgpu::Buffer>,
    next_u: usize,
    /// Placeholders for unused bindings.
    dummy_tex: wgpu::TextureView,
    dummy_depth: wgpu::TextureView,
    /// The shadow map's comparison sampler.
    cmp: wgpu::Sampler,
    /// Bloom pyramid, for the size it was made for, and AO buffers (the AO
    /// and its view depth).
    bloom: Vec<(wgpu::Texture, u32, u32)>,
    ao: Vec<wgpu::Texture>,
    ao_depth: Option<wgpu::Texture>,
    size: (u32, u32),
}

/// The main target's textures the passes use.
pub struct Frame<'a> {
    pub color: &'a wgpu::Texture,
    pub snapshot: &'a wgpu::Texture,
    pub depth: &'a wgpu::Texture,
    pub zsnap: &'a wgpu::Texture,
    pub w: u32,
    pub h: u32,
}

impl Post {
    pub fn new(dev: &wgpu::Device) -> Post {
        let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssx3-gfx post"),
            source: wgpu::ShaderSource::Wgsl(include_str!("post.wgsl").into()),
        });
        let tex_entry = |binding, sample_type| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let layout = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssx3-gfx post"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                tex_entry(1, wgpu::TextureSampleType::Float { filterable: true }),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                tex_entry(3, wgpu::TextureSampleType::Depth),
                tex_entry(4, wgpu::TextureSampleType::Float { filterable: true }),
                // The shadow map and its comparison sampler.
                tex_entry(5, wgpu::TextureSampleType::Depth),
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
                // The AO's view depth.
                tex_entry(7, wgpu::TextureSampleType::Float { filterable: false }),
            ],
        });
        let pipe_layout = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ssx3-gfx post"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let samp = dev.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ssx3-gfx post"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let ubufs = (0..UNIFORMS)
            .map(|_| {
                dev.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("ssx3-gfx post uniforms"),
                    size: U_BYTES as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();
        let mk = |format, usage| {
            dev.create_texture(&wgpu::TextureDescriptor {
                label: Some("ssx3-gfx post dummy"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
            .create_view(&Default::default())
        };
        let dummy_tex = mk(
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let dummy_depth = mk(
            crate::draw::DEPTH_FORMAT,
            wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let cmp = dev.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ssx3-gfx shadow compare"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        Post {
            module,
            layout,
            pipe_layout,
            pipes: HashMap::new(),
            samp,
            ubufs,
            next_u: 0,
            dummy_tex,
            dummy_depth,
            cmp,
            bloom: Vec::new(),
            ao: Vec::new(),
            ao_depth: None,
            size: (0, 0),
        }
    }

    /// Start a frame (the uniform buffers are reused per frame).
    pub fn begin_frame(&mut self) {
        self.next_u = 0;
    }

    fn pipe(
        &mut self,
        dev: &wgpu::Device,
        entry: &'static str,
        format: wgpu::TextureFormat,
        write: Write,
    ) -> wgpu::RenderPipeline {
        let key = (entry, format, write);
        if let Some(p) = self.pipes.get(&key) {
            return p.clone();
        }
        let (blend, mask) = match write {
            Write::Rgb => (None, wgpu::ColorWrites::COLOR),
            Write::All => (None, wgpu::ColorWrites::ALL),
            Write::Add => (
                Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent::REPLACE,
                }),
                wgpu::ColorWrites::ALL,
            ),
            Write::Multiply => (
                Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::Zero,
                        dst_factor: wgpu::BlendFactor::Src,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::Zero,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                }),
                wgpu::ColorWrites::COLOR,
            ),
        };
        let p = dev.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(entry),
            layout: Some(&self.pipe_layout),
            vertex: wgpu::VertexState {
                module: &self.module,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &self.module,
                entry_point: Some(entry),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend,
                    write_mask: mask,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        self.pipes.insert(key, p.clone());
        p
    }

    /// One full-screen pass.
    #[allow(clippy::too_many_arguments)]
    fn pass(
        &mut self,
        dev: &wgpu::Device,
        queue: &wgpu::Queue,
        enc: &mut wgpu::CommandEncoder,
        entry: &'static str,
        target: &wgpu::Texture,
        write: Write,
        src: Option<&wgpu::TextureView>,
        depth: Option<&wgpu::TextureView>,
        image: Option<&wgpu::TextureView>,
        u: U,
    ) {
        self.pass_with(
            dev, queue, enc, entry, target, write, src, depth, image, None, None, u,
        )
    }

    /// [`Post::pass`] with the shadow map or the AO's view depth bound.
    #[allow(clippy::too_many_arguments)]
    fn pass_with(
        &mut self,
        dev: &wgpu::Device,
        queue: &wgpu::Queue,
        enc: &mut wgpu::CommandEncoder,
        entry: &'static str,
        target: &wgpu::Texture,
        write: Write,
        src: Option<&wgpu::TextureView>,
        depth: Option<&wgpu::TextureView>,
        image: Option<&wgpu::TextureView>,
        smap: Option<&wgpu::TextureView>,
        ao_depth: Option<&wgpu::TextureView>,
        u: U,
    ) {
        let pipe = self.pipe(dev, entry, target.format(), write);
        if self.next_u >= self.ubufs.len() {
            return;
        }
        let ub = &self.ubufs[self.next_u];
        self.next_u += 1;
        queue.write_buffer(ub, 0, &u.bytes());
        let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssx3-gfx post"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ub.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(src.unwrap_or(&self.dummy_tex)),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.samp),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(
                        depth.unwrap_or(&self.dummy_depth),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(image.unwrap_or(&self.dummy_tex)),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(smap.unwrap_or(&self.dummy_depth)),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(&self.cmp),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(
                        ao_depth.unwrap_or(&self.dummy_tex),
                    ),
                },
            ],
        });
        let view = target.create_view(&Default::default());
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(entry),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: if write == Write::All {
                        wgpu::LoadOp::Clear(wgpu::Color::BLACK)
                    } else {
                        wgpu::LoadOp::Load
                    },
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipe);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }

    /// The bloom pyramid for a `w`×`h` image.
    fn ensure(&mut self, dev: &wgpu::Device, w: u32, h: u32) {
        if self.size == (w, h) {
            return;
        }
        self.size = (w, h);
        self.bloom = (1..=BLOOM_LEVELS)
            .map(|i| {
                let (bw, bh) = ((w >> i).max(1), (h >> i).max(1));
                (target(dev, bw, bh, BLOOM_FORMAT, "ssx3-gfx bloom"), bw, bh)
            })
            .collect();
    }

    fn copy(enc: &mut wgpu::CommandEncoder, from: &wgpu::Texture, to: &wgpu::Texture) {
        enc.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: from,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: to,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: from.width(),
                height: from.height(),
                depth_or_array_layers: 1,
            },
        );
    }

    /// Ambient occlusion onto the image (before the game's post passes),
    /// computed at half resolution and upsampled along depth edges.
    #[allow(clippy::too_many_arguments)]
    pub fn ao(
        &mut self,
        dev: &wgpu::Device,
        queue: &wgpu::Queue,
        enc: &mut wgpu::CommandEncoder,
        f: &Frame,
        cam: Option<&crate::gs::Cam>,
        fxy: (f32, f32),
    ) {
        let stride = 2;
        let (aw, ah) = (f.w.div_ceil(stride), f.h.div_ceil(stride));
        let made =
            |t: &wgpu::Texture, w: u32, h: u32| (t.width(), t.height()) == (w.max(1), h.max(1));
        if !self.ao.first().is_some_and(|t| made(t, aw, ah)) {
            self.ao = (0..2)
                .map(|_| target(dev, aw, ah, AO_FORMAT, "ssx3-gfx ao"))
                .collect();
        }
        if !self.ao_depth.as_ref().is_some_and(|t| made(t, f.w, f.h)) {
            self.ao_depth = Some(target(dev, f.w, f.h, AO_DEPTH_FORMAT, "ssx3-gfx ao depth"));
        }
        // With the terrain camera, exact view depth (else 1 / z).
        let mut u = U {
            size: [f.w as f32, f.h as f32],
            strength: 0.75,
            stride,
            ..Default::default()
        };
        let rays = cam.and_then(|c| Some((c, cam_matrices(c)?, ao_rays(c, fxy)?)));
        if let Some((c, (w2s, s2w, _), ray)) = rays {
            u.w2s = w2s;
            u.s2w = s2w;
            u.ray = ray;
            u.off = [c.off[0], c.off[1], fxy.0, fxy.1];
            u.zp = [
                c.zmax as f32 + 1.0,
                if c.xyzf { 1.0 } else { 16.0 },
                0.0,
                0.0,
            ];
        }
        // (No render pass has the depth attached here: it is read as is.)
        let zv = f.depth.create_view(&Default::default());
        let lin = self.ao_depth.clone().expect("made above");
        let lv = lin.create_view(&Default::default());
        let (a0, a1) = (self.ao[0].clone(), self.ao[1].clone());
        let (v0, v1) = (
            a0.create_view(&Default::default()),
            a1.create_view(&Default::default()),
        );
        // The upsample pass multiplies its result onto the image.
        let apply = if show("ao") {
            Write::Rgb
        } else {
            Write::Multiply
        };
        self.pass(
            dev,
            queue,
            enc,
            "fs_ao_depth",
            &lin,
            Write::All,
            None,
            Some(&zv),
            None,
            u,
        );
        self.pass_with(
            dev,
            queue,
            enc,
            "fs_ao",
            &a0,
            Write::All,
            None,
            Some(&zv),
            None,
            None,
            Some(&lv),
            u,
        );
        self.pass_with(
            dev,
            queue,
            enc,
            "fs_ao_blur",
            &a1,
            Write::All,
            Some(&v0),
            None,
            None,
            None,
            Some(&lv),
            U {
                dir: [1.0, 0.0],
                ..u
            },
        );
        self.pass_with(
            dev,
            queue,
            enc,
            "fs_ao_blur",
            &a0,
            Write::All,
            Some(&v1),
            None,
            None,
            None,
            Some(&lv),
            U {
                dir: [0.0, 1.0],
                ..u
            },
        );
        self.pass_with(
            dev,
            queue,
            enc,
            "fs_ao_up",
            f.color,
            apply,
            Some(&v0),
            Some(&zv),
            None,
            None,
            Some(&lv),
            u,
        );
    }

    /// Sun shadows onto the image (before the game's post passes): `smap`
    /// drawn from `light` (shadow.rs).
    #[allow(clippy::too_many_arguments)]
    pub fn shadow(
        &mut self,
        dev: &wgpu::Device,
        queue: &wgpu::Queue,
        enc: &mut wgpu::CommandEncoder,
        f: &Frame,
        cam: &crate::gs::Cam,
        fxy: (f32, f32),
        smap: &wgpu::TextureView,
        light: &crate::shadow::Light,
    ) {
        self.ensure(dev, f.w, f.h);
        let Some((w2s, s2w, _)) = cam_matrices(cam) else {
            return;
        };
        let t = crate::shadow::tuning();
        let u = U {
            size: [f.w as f32, f.h as f32],
            w2s,
            s2w,
            off: [cam.off[0], cam.off[1], fxy.0, fxy.1],
            zp: [
                cam.zmax as f32 + 1.0,
                if cam.xyzf { 1.0 } else { 16.0 },
                0.0,
                0.0,
            ],
            light: light.lights[shadow_cascade()],
            sun: [light.sun[0], light.sun[1], light.sun[2], t.strength],
            sp: [
                light.texel,
                light.world_texel[shadow_cascade()] * 1.5,
                debug_view(),
                0.0,
            ],
            ..Default::default()
        };
        Self::copy(enc, f.depth, f.zsnap);
        let zv = f.zsnap.create_view(&Default::default());
        let w = Write::Rgb;
        self.pass_with(
            dev,
            queue,
            enc,
            "fs_shadow_apply",
            f.color,
            w,
            None,
            Some(&zv),
            None,
            Some(smap),
            None,
            u,
        );
    }

    /// Modern bloom and FXAA (before the HUD).
    pub fn before_hud(
        &mut self,
        dev: &wgpu::Device,
        queue: &wgpu::Queue,
        enc: &mut wgpu::CommandEncoder,
        f: &Frame,
        o: &Options,
    ) {
        self.ensure(dev, f.w, f.h);
        let size = [f.w as f32, f.h as f32];
        let snap = f.snapshot.create_view(&Default::default());
        if o.bloom == Bloom::Modern {
            Self::copy(enc, f.color, f.snapshot);
            let levels: Vec<(wgpu::Texture, u32, u32)> = self.bloom.clone();
            let views: Vec<wgpu::TextureView> = levels
                .iter()
                .map(|(t, _, _)| t.create_view(&Default::default()))
                .collect();
            let texel = |w: u32, h: u32| [1.0 / w as f32, 1.0 / h as f32];
            self.pass(
                dev,
                queue,
                enc,
                "fs_bright",
                &levels[0].0,
                Write::All,
                Some(&snap),
                None,
                None,
                U {
                    texel: texel(f.w, f.h),
                    size,
                    ..Default::default()
                },
            );
            for i in 0..BLOOM_LEVELS - 1 {
                let (_, w, h) = levels[i];
                self.pass(
                    dev,
                    queue,
                    enc,
                    "fs_down",
                    &levels[i + 1].0,
                    Write::All,
                    Some(&views[i]),
                    None,
                    None,
                    U {
                        texel: texel(w, h),
                        size,
                        ..Default::default()
                    },
                );
            }
            for i in (0..BLOOM_LEVELS - 1).rev() {
                let (_, w, h) = levels[i + 1];
                self.pass(
                    dev,
                    queue,
                    enc,
                    "fs_up",
                    &levels[i].0,
                    Write::Add,
                    Some(&views[i + 1]),
                    None,
                    None,
                    U {
                        texel: texel(w, h),
                        size,
                        ..Default::default()
                    },
                );
            }
            self.pass(
                dev,
                queue,
                enc,
                "fs_bloom_mix",
                f.color,
                Write::Rgb,
                Some(&views[0]),
                None,
                Some(&snap),
                U {
                    strength: 0.2,
                    size,
                    ..Default::default()
                },
            );
        }
        if o.aa == Aa::Fxaa {
            Self::copy(enc, f.color, f.snapshot);
            self.pass(
                dev,
                queue,
                enc,
                "fs_fxaa",
                f.color,
                Write::Rgb,
                Some(&snap),
                None,
                None,
                U {
                    texel: [1.0 / f.w as f32, 1.0 / f.h as f32],
                    size,
                    ..Default::default()
                },
            );
        }
    }
}

/// A `w`×`h` texture the passes draw into and read.
fn target(
    dev: &wgpu::Device,
    w: u32,
    h: u32,
    format: wgpu::TextureFormat,
    label: &'static str,
) -> wgpu::Texture {
    dev.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: w.max(1),
            height: h.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

/// World → GS screen (homogeneous, rows) of a terrain camera:
/// screen = gO·w + gS ∘ (M·p).
pub(crate) fn screen_matrix(c: &crate::gs::Cam) -> [[f64; 4]; 4] {
    let m = |r: usize, k: usize| c.m[k][r] as f64;
    let mut s = [[0f64; 4]; 4];
    for r in 0..4 {
        for k in 0..4 {
            s[r][k] = if r < 3 {
                c.gscale[r] as f64 * m(r, k) + c.goffset[r] as f64 * m(3, k)
            } else {
                m(3, k)
            };
        }
    }
    s
}

pub(crate) fn mul4(m: &[[f64; 4]; 4], v: [f64; 4]) -> [f64; 4] {
    std::array::from_fn(|r| (0..4).map(|k| m[r][k] * v[k]).sum())
}

/// The world point of GS screen point (x, y, screen z).
pub(crate) fn unproject(inv: &[[f64; 4]; 4], x: f64, y: f64, z: f64) -> [f64; 3] {
    let h = mul4(inv, [x, y, z, 1.0]);
    [h[0] / h[3], h[1] / h[3], h[2] / h[3]]
}

/// The camera's world → screen and screen → world matrices (columns, for
/// WGSL), in world coordinates relative to a point in front of the camera:
/// the game's world coordinates are large (≈ 10⁵), too large for f32
/// reconstruction. Also that point (world).
pub(crate) fn cam_matrices(c: &crate::gs::Cam) -> Option<([[f32; 4]; 4], [[f32; 4]; 4], [f64; 3])> {
    let (s2, inv2, o) = rebased(c)?;
    let cols = |a: [[f64; 4]; 4]| -> [[f32; 4]; 4] {
        std::array::from_fn(|k| std::array::from_fn(|r| a[r][k] as f32))
    };
    Some((cols(s2), cols(inv2), o))
}

/// [`cam_matrices`] in rows and f64.
fn rebased(c: &crate::gs::Cam) -> Option<([[f64; 4]; 4], [[f64; 4]; 4], [f64; 3])> {
    let s = screen_matrix(c);
    let inv = invert(s)?;
    let k = if c.xyzf { 1.0 } else { 16.0 };
    let o = unproject(&inv, 2048.0, 2048.0, c.zmax as f64 * 0.5 / k);
    // S' = S · T(o), inv' = T(−o) · inv.
    let mut s2 = s;
    for row in s2.iter_mut() {
        row[3] += row[0] * o[0] + row[1] * o[1] + row[2] * o[2];
    }
    let mut inv2 = inv;
    for c4 in 0..4 {
        for r in 0..3 {
            inv2[r][c4] -= o[r] * inv[3][c4];
        }
    }
    Some((s2, inv2, o))
}

/// The terrain camera's rays for the AO (columns, `ray` in post.wgsl), in
/// [`cam_matrices`]' coordinates: the camera, then the offset from it per
/// unit of view depth along target pixel (0, 0)'s ray and that offset's
/// change per target pixel in x and y. The point at view depth `w` on the
/// ray through target point `q` is `camera + w · (d0 + q.x · dx + q.y · dy)`
/// (exact for a perspective camera: both depend linearly on the screen
/// position at a fixed view depth). `fxy`: target pixels per GS pixel.
pub(crate) fn ao_rays(c: &crate::gs::Cam, fxy: (f32, f32)) -> Option<[[f32; 4]; 4]> {
    let (s, inv, _) = rebased(c)?;
    // The camera: the point the screen's x, y and w all vanish at.
    let h = mul4(&inv, [0.0, 0.0, 1.0, 0.0]);
    if h[3].abs() < 1e-12 {
        return None;
    }
    let cam = [h[0] / h[3], h[1] / h[3], h[2] / h[3]];
    let z = c.zmax as f64 * 0.5 / if c.xyzf { 1.0 } else { 16.0 };
    // The offset per unit view depth along GS screen point (x, y)'s ray.
    let per_w = |x: f64, y: f64| -> Option<[f64; 3]> {
        let p = unproject(&inv, x, y, z);
        let w = mul4(&s, [p[0], p[1], p[2], 1.0])[3];
        (w.abs() > 1e-12).then(|| std::array::from_fn(|i| (p[i] - cam[i]) / w))
    };
    // Target pixel q is GS screen point q / fxy - 0.5 + XYOFFSET.
    let (x0, y0) = (c.off[0] as f64 - 0.5, c.off[1] as f64 - 0.5);
    let d0 = per_w(x0, y0)?;
    let dx = per_w(x0 + 1.0, y0)?;
    let dy = per_w(x0, y0 + 1.0)?;
    let col = |v: [f64; 3], k: f64| [v[0] as f32, v[1] as f32, v[2] as f32, k as f32];
    let step = |a: [f64; 3], f: f32| col(std::array::from_fn(|i| (a[i] - d0[i]) / f as f64), 0.0);
    Some([
        col(cam, 1.0),
        col(d0, 0.0),
        step(dx, fxy.0),
        step(dy, fxy.1),
    ])
}

/// The inverse of a 4×4 matrix (rows), by Gauss-Jordan elimination.
pub(crate) fn invert(a: [[f64; 4]; 4]) -> Option<[[f64; 4]; 4]> {
    let mut m = a;
    let mut inv = [[0f64; 4]; 4];
    for (i, row) in inv.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for c in 0..4 {
        let p = (c..4).max_by(|&x, &y| m[x][c].abs().total_cmp(&m[y][c].abs()))?;
        if m[p][c].abs() < 1e-12 {
            return None;
        }
        m.swap(c, p);
        inv.swap(c, p);
        let d = m[c][c];
        for k in 0..4 {
            m[c][k] /= d;
            inv[c][k] /= d;
        }
        for r in 0..4 {
            if r != c {
                let f = m[r][c];
                for k in 0..4 {
                    m[r][k] -= f * m[c][k];
                    inv[r][k] -= f * inv[c][k];
                }
            }
        }
    }
    Some(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_round_trip() {
        let a = [
            [2.0, 0.0, 0.0, 1.0],
            [0.0, 3.0, 0.5, 0.0],
            [0.1, 0.0, 1.0, 4.0],
            [0.0, 0.0, 1.0, 0.0],
        ];
        let b = invert(a).unwrap();
        for r in 0..4 {
            for c in 0..4 {
                let v: f64 = (0..4).map(|k| a[r][k] * b[k][c]).sum();
                assert!((v - if r == c { 1.0 } else { 0.0 }).abs() < 1e-9);
            }
        }
    }
    #[test]
    fn ao_rays_rebuild_unprojected_points() {
        // A perspective camera (rows: x, y, 0.5 z + 10, z), its guard
        // mapping onto a 4096-wide screen, a 2× target.
        let mut m = [[0f32; 4]; 4];
        m[0][0] = 1.0;
        m[1][1] = 1.0;
        m[2][2] = 0.5;
        m[2][3] = 1.0;
        m[3][2] = 10.0;
        let c = crate::gs::Cam {
            m,
            gscale: [500.0, -500.0, 1.0e6, 1.0],
            goffset: [2048.0, 2048.0, 0.0, 0.0],
            off: [1728.0, 1824.0],
            zmax: 0xff_ffff,
            xyzf: false,
        };
        let fxy = (2.0, 2.0);
        let (s, inv, _) = rebased(&c).unwrap();
        let r = ao_rays(&c, fxy).unwrap();
        for (qx, qy) in [(0.5, 0.5), (640.5, 448.5), (1279.5, 3.5)] {
            let (x, y) = (
                qx / fxy.0 as f64 - 0.5 + 1728.0,
                qy / fxy.1 as f64 - 0.5 + 1824.0,
            );
            for z in [1.0e5, 5.0e5, 9.0e5] {
                let pt = unproject(&inv, x, y, z);
                let w = mul4(&s, [pt[0], pt[1], pt[2], 1.0])[3];
                for i in 0..3 {
                    let d = r[1][i] as f64 + qx * r[2][i] as f64 + qy * r[3][i] as f64;
                    let got = r[0][i] as f64 + w * d;
                    assert!(
                        (got - pt[i]).abs() < 1e-3 * (1.0 + pt[i].abs()),
                        "{qx} {qy} {z}: {got} vs {}",
                        pt[i]
                    );
                }
            }
        }
    }
}
