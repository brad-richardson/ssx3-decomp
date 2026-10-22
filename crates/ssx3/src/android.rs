//! Android input: game controllers and keyboards, read from the activity's
//! input queue.
//!
//! winit 0.30 passes an Android controller on only as touches with its X/Y
//! axes (the right stick, the triggers and the hat are lost) and its buttons
//! as unidentified keys, so the window reads the queue itself at the start of
//! every event-loop iteration, before winit does (`App::new_events`).
//!
//! Buttons map by position as on the desktop (Android names them after the
//! Xbox layout): the bottom face button is ✕, the right one ○, the left one
//! □, the top one △. Analog triggers press L2/R2 past half travel; a hat
//! (d-pad reported as axes) presses the d-pad. A keyboard has the desktop's
//! keys (`pad_bit` in `app.rs`; the arrows arrive as d-pad keys). The first
//! controller drives pad 1 together with a keyboard, a second one pad 2
//! (plugged in with `SSX3_PAD2=1`). The controller's home button (or F10, or
//! Back) opens the settings overlay.
//!
//! The mapping works on Android's key codes and axis values as plain numbers
//! and is tested on the host.

use crate::app::MenuKey;

// android.view.KeyEvent key codes.
const KEYCODE_BACK: u32 = 4;
const KEYCODE_1: u32 = 8;
const KEYCODE_2: u32 = 9;
const KEYCODE_DPAD_UP: u32 = 19;
const KEYCODE_DPAD_DOWN: u32 = 20;
const KEYCODE_DPAD_LEFT: u32 = 21;
const KEYCODE_DPAD_RIGHT: u32 = 22;
const KEYCODE_DPAD_CENTER: u32 = 23;
const KEYCODE_A: u32 = 29;
const KEYCODE_Q: u32 = 45;
const KEYCODE_S: u32 = 47;
const KEYCODE_W: u32 = 51;
const KEYCODE_X: u32 = 52;
const KEYCODE_Z: u32 = 54;
const KEYCODE_SHIFT_RIGHT: u32 = 60;
const KEYCODE_SPACE: u32 = 62;
const KEYCODE_ENTER: u32 = 66;
const KEYCODE_DEL: u32 = 67;
const KEYCODE_BUTTON_A: u32 = 96;
const KEYCODE_BUTTON_B: u32 = 97;
const KEYCODE_BUTTON_X: u32 = 99;
const KEYCODE_BUTTON_Y: u32 = 100;
const KEYCODE_BUTTON_L1: u32 = 102;
const KEYCODE_BUTTON_R1: u32 = 103;
const KEYCODE_BUTTON_L2: u32 = 104;
const KEYCODE_BUTTON_R2: u32 = 105;
const KEYCODE_BUTTON_THUMBL: u32 = 106;
const KEYCODE_BUTTON_THUMBR: u32 = 107;
const KEYCODE_BUTTON_START: u32 = 108;
const KEYCODE_BUTTON_SELECT: u32 = 109;
const KEYCODE_BUTTON_MODE: u32 = 110;
const KEYCODE_ESCAPE: u32 = 111;
const KEYCODE_F10: u32 = 140;

// android.view.InputDevice sources: the class bits a controller's events
// carry.
const SOURCE_GAMEPAD: u32 = 0x0000_0401;
const SOURCE_JOYSTICK: u32 = 0x0100_0010;

/// libpad button bit for an Android key code: a controller's buttons and
/// d-pad, and the desktop's keyboard keys.
pub fn button_bit(keycode: u32) -> Option<u16> {
    Some(match keycode {
        KEYCODE_BUTTON_SELECT | KEYCODE_SHIFT_RIGHT => 1 << 0,
        KEYCODE_BUTTON_THUMBL => 1 << 1,
        KEYCODE_BUTTON_THUMBR => 1 << 2,
        KEYCODE_BUTTON_START | KEYCODE_ENTER => 1 << 3,
        KEYCODE_DPAD_UP => 1 << 4,
        KEYCODE_DPAD_RIGHT => 1 << 5,
        KEYCODE_DPAD_DOWN => 1 << 6,
        KEYCODE_DPAD_LEFT => 1 << 7,
        KEYCODE_BUTTON_L2 | KEYCODE_1 => 1 << 8,
        KEYCODE_BUTTON_R2 | KEYCODE_2 => 1 << 9,
        KEYCODE_BUTTON_L1 | KEYCODE_Q => 1 << 10,
        KEYCODE_BUTTON_R1 | KEYCODE_W => 1 << 11,
        KEYCODE_BUTTON_Y | KEYCODE_A => 1 << 12, // Triangle (top)
        KEYCODE_BUTTON_B | KEYCODE_S => 1 << 13, // Circle (right)
        KEYCODE_BUTTON_A | KEYCODE_X => 1 << 14, // Cross (bottom)
        KEYCODE_BUTTON_X | KEYCODE_Z => 1 << 15, // Square (left)
        _ => return None,
    })
}

/// Whether a key opens the settings overlay.
pub fn opens_menu(keycode: u32) -> bool {
    matches!(keycode, KEYCODE_BUTTON_MODE | KEYCODE_F10 | KEYCODE_BACK)
}

/// Overlay navigation while it is open.
pub fn menu_key(keycode: u32) -> Option<MenuKey> {
    Some(match keycode {
        KEYCODE_DPAD_UP => MenuKey::Up,
        KEYCODE_DPAD_DOWN => MenuKey::Down,
        KEYCODE_DPAD_LEFT => MenuKey::Left,
        KEYCODE_DPAD_RIGHT => MenuKey::Right,
        KEYCODE_BUTTON_A | KEYCODE_X | KEYCODE_ENTER | KEYCODE_SPACE | KEYCODE_DPAD_CENTER => MenuKey::Select,
        KEYCODE_BUTTON_B | KEYCODE_BUTTON_START | KEYCODE_BUTTON_MODE | KEYCODE_F10 | KEYCODE_BACK | KEYCODE_ESCAPE | KEYCODE_DEL => MenuKey::Close,
        _ => return None,
    })
}

/// Whether an input source (a bit set) is a game controller.
pub fn is_controller(source: u32) -> bool {
    source & SOURCE_GAMEPAD == SOURCE_GAMEPAD || source & SOURCE_JOYSTICK == SOURCE_JOYSTICK
}

/// A stick axis (-1..1; Android: right and down positive) as the PS2's byte
/// (0 = left/up, 0xFF = right/down).
pub fn stick_byte(v: f32) -> u8 {
    ((v.clamp(-1.0, 1.0) * 127.5) + 127.5).round() as u8
}

/// The buttons a controller reports as axes: the hat (d-pad: -1, 0 or 1)
/// and the analog triggers (0..1; L2/R2 past half travel).
pub fn axis_buttons(hat_x: f32, hat_y: f32, l2: f32, r2: f32) -> u16 {
    let mut b = 0;
    for (on, bit) in [(hat_y < -0.5, 1 << 4), (hat_x > 0.5, 1 << 5), (hat_y > 0.5, 1 << 6), (hat_x < -0.5, 1 << 7), (l2 > 0.5, 1 << 8), (r2 > 0.5, 1 << 9)] {
        if on {
            b |= bit;
        }
    }
    b
}

/// Pads 1 and 2 as the Android input events left them.
pub struct Pads {
    /// Controllers by device id, in the order they were first used: the
    /// index is the port.
    devices: Vec<i32>,
    /// Buttons held, from key events.
    keys: [u16; 2],
    /// Buttons from axes (hat, analog triggers).
    axes: [u16; 2],
    /// Right X, right Y, left X, left Y.
    sticks: [[u8; 4]; 2],
}

impl Pads {
    pub fn new() -> Pads {
        Pads { devices: Vec::new(), keys: [0; 2], axes: [0; 2], sticks: [[0x80; 4]; 2] }
    }

    pub fn buttons(&self, port: usize) -> u16 {
        self.keys[port] | self.axes[port]
    }

    pub fn sticks(&self, port: usize) -> [u8; 4] {
        self.sticks[port]
    }

    /// The port of an input source: a controller's by its device (the first
    /// two seen get pads 1 and 2), anything else pad 1.
    fn port(&mut self, source: u32, device: i32) -> usize {
        if !is_controller(source) {
            return 0;
        }
        match self.devices.iter().position(|d| *d == device) {
            Some(p) => p,
            None if self.devices.len() < 2 => {
                self.devices.push(device);
                self.devices.len() - 1
            }
            None => 0,
        }
    }
}

#[cfg(target_os = "android")]
impl Pads {
    /// Read every pending input event. Returns the keys pressed (key codes,
    /// repeats left out) for the overlay; while it is open (`menu`) presses
    /// do not reach the pads.
    pub fn read_input(&mut self, app: &winit::platform::android::activity::AndroidApp, menu: bool) -> Vec<u32> {
        use winit::platform::android::activity::input::{Axis, InputEvent, KeyAction};
        use winit::platform::android::activity::InputStatus;
        let mut pressed = Vec::new();
        let mut iter = match app.input_events_iter() {
            Ok(iter) => iter,
            Err(e) => {
                eprintln!("[pad] no input events: {e:?}");
                return pressed;
            }
        };
        while iter.next(|event| match event {
            InputEvent::KeyEvent(key) => {
                let code: u32 = key.key_code().into();
                let down = match key.action() {
                    KeyAction::Down => true,
                    KeyAction::Up => false,
                    _ => return InputStatus::Unhandled,
                };
                if down && key.repeat_count() == 0 && (opens_menu(code) || menu_key(code).is_some()) {
                    pressed.push(code);
                }
                match button_bit(code) {
                    Some(bit) => {
                        let port = self.port(key.source().into(), key.device_id());
                        if !down {
                            self.keys[port] &= !bit;
                        } else if !menu {
                            self.keys[port] |= bit;
                        }
                        InputStatus::Handled
                    }
                    // Volume and the rest stay with the system.
                    None if opens_menu(code) || menu_key(code).is_some() => InputStatus::Handled,
                    None => InputStatus::Unhandled,
                }
            }
            InputEvent::MotionEvent(motion) if is_controller(motion.source().into()) => {
                let port = self.port(motion.source().into(), motion.device_id());
                let p = motion.pointer_at_index(0);
                let a = |axis| p.axis_value(axis);
                self.sticks[port] = [stick_byte(a(Axis::Z)), stick_byte(a(Axis::Rz)), stick_byte(a(Axis::X)), stick_byte(a(Axis::Y))];
                self.axes[port] = axis_buttons(a(Axis::HatX), a(Axis::HatY), a(Axis::Ltrigger).max(a(Axis::Brake)), a(Axis::Rtrigger).max(a(Axis::Gas)));
                InputStatus::Handled
            }
            _ => InputStatus::Unhandled,
        }) {}
        pressed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn face_buttons_by_position() {
        assert_eq!(button_bit(KEYCODE_BUTTON_A), Some(1 << 14)); // bottom: Cross
        assert_eq!(button_bit(KEYCODE_BUTTON_B), Some(1 << 13)); // right: Circle
        assert_eq!(button_bit(KEYCODE_BUTTON_X), Some(1 << 15)); // left: Square
        assert_eq!(button_bit(KEYCODE_BUTTON_Y), Some(1 << 12)); // top: Triangle
    }

    #[test]
    fn every_pad_button_once() {
        let pad = [
            KEYCODE_BUTTON_SELECT, KEYCODE_BUTTON_THUMBL, KEYCODE_BUTTON_THUMBR, KEYCODE_BUTTON_START,
            KEYCODE_DPAD_UP, KEYCODE_DPAD_RIGHT, KEYCODE_DPAD_DOWN, KEYCODE_DPAD_LEFT,
            KEYCODE_BUTTON_L2, KEYCODE_BUTTON_R2, KEYCODE_BUTTON_L1, KEYCODE_BUTTON_R1,
            KEYCODE_BUTTON_Y, KEYCODE_BUTTON_B, KEYCODE_BUTTON_A, KEYCODE_BUTTON_X,
        ];
        // In libpad bit order: all 16 bits, each once.
        for (i, k) in pad.into_iter().enumerate() {
            assert_eq!(button_bit(k), Some(1 << i), "key code {k}");
        }
        // The keyboard has the same 16 (as on the desktop).
        let keys = [
            KEYCODE_SHIFT_RIGHT, KEYCODE_ENTER, KEYCODE_1, KEYCODE_2, KEYCODE_Q, KEYCODE_W,
            KEYCODE_A, KEYCODE_S, KEYCODE_X, KEYCODE_Z,
        ];
        let bits = keys.into_iter().fold(0, |b, k| b | button_bit(k).unwrap());
        assert_eq!(bits, 0xff09);
        // Volume stays with the system.
        assert_eq!(button_bit(24), None);
        assert!(menu_key(24).is_none() && !opens_menu(24));
    }

    #[test]
    fn sticks_centre_and_ends() {
        assert_eq!(stick_byte(0.0), 0x80);
        assert_eq!(stick_byte(-1.0), 0x00);
        assert_eq!(stick_byte(1.0), 0xff);
        assert_eq!(stick_byte(-2.0), 0x00);
    }

    #[test]
    fn hat_and_triggers() {
        assert_eq!(axis_buttons(0.0, 0.0, 0.0, 0.0), 0);
        assert_eq!(axis_buttons(0.0, -1.0, 0.0, 0.0), 1 << 4); // up
        assert_eq!(axis_buttons(1.0, 1.0, 0.0, 0.0), 1 << 5 | 1 << 6); // right + down
        assert_eq!(axis_buttons(-1.0, 0.0, 0.0, 0.0), 1 << 7); // left
        assert_eq!(axis_buttons(0.0, 0.0, 0.4, 0.6), 1 << 9);
        assert_eq!(axis_buttons(0.0, 0.0, 1.0, 0.0), 1 << 8);
    }

    #[test]
    fn controllers_get_ports_in_order() {
        let gamepad = SOURCE_GAMEPAD | 0x100; // key events: keyboard | gamepad
        let mut p = Pads::new();
        assert_eq!((p.buttons(1), p.sticks(1)), (0, [0x80; 4]));
        assert_eq!(p.port(0x101, 3), 0); // a keyboard: pad 1
        assert_eq!(p.port(gamepad, 7), 0);
        assert_eq!(p.port(SOURCE_JOYSTICK, 9), 1);
        assert_eq!(p.port(gamepad, 7), 0);
        assert_eq!(p.port(gamepad, 11), 0); // a third: pad 1
        assert_eq!(p.port(SOURCE_JOYSTICK, 9), 1);
    }

    #[test]
    fn overlay_keys() {
        assert!(opens_menu(KEYCODE_BUTTON_MODE) && opens_menu(KEYCODE_BACK));
        assert!(!opens_menu(KEYCODE_BUTTON_START));
        assert_eq!(menu_key(KEYCODE_BUTTON_A), Some(MenuKey::Select));
        assert_eq!(menu_key(KEYCODE_BUTTON_B), Some(MenuKey::Close));
        assert_eq!(menu_key(KEYCODE_DPAD_LEFT), Some(MenuKey::Left));
    }
}
