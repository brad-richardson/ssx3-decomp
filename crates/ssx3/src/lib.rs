//! SSX 3 (USA) on Android: the window (`app.rs`) as the library
//! `android.app.NativeActivity` loads (`libssx3.so`; README "Android").
//! Natively this is an empty library: the desktop runs `main.rs`.
//!
//! Without a working directory of its own, the app keeps its files (disc
//! image, memory card, settings, texture packs, screenshots) in its external
//! files directory (`/sdcard/Android/data/<package>/files`).
#![cfg(target_os = "android")]

// The generated EE code in the chosen form (features `ee-threads` / `ee-unwind`).
#[cfg(feature = "ee-unwind")]
extern crate ssx3_ee_unwind as ssx3_ee;
#[cfg(all(feature = "ee-threads", not(feature = "ee-unwind")))]
extern crate ssx3_ee_threads as ssx3_ee;

mod android;
mod app;
mod game;
mod png;
mod present;
mod settings;

use winit::event_loop::EventLoop;
use winit::platform::android::activity::AndroidApp;
use winit::platform::android::EventLoopBuilderExtAndroid;

/// The entry point the NativeActivity glue (android-activity) calls on the
/// app's main thread.
#[no_mangle]
fn android_main(android: AndroidApp) {
    if let Some(dir) = android.external_data_path().or_else(|| android.internal_data_path()) {
        ssx3_core::data_dir::set(dir);
    }
    let mut event_loop = EventLoop::builder();
    event_loop.with_android_app(android);
    app::run(event_loop);
}
