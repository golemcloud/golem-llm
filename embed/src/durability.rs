use crate::embed::Guest;

#[cfg(feature = "durability")]
pub trait DurableEmbed {
    /// Saves the state of the embedding provider to Golem's durability API.
    fn save_state(&self) -> Result<(), String>;

    /// Loads the state of the embedding provider from Golem's durability API.
    fn load_state(&self) -> Result<(), String>;
}

#[cfg(feature = "durability")]
pub trait ExtendedGuest: Guest {
    /// Saves the state of the embedding provider to Golem's durability API.
    fn save_state(&self) -> Result<(), String>;

    /// Loads the state of the embedding provider from Golem's durability API.
    fn load_state(&self) -> Result<(), String>;
}

#[cfg(feature = "durability")]
impl<T: DurableEmbed> ExtendedGuest for T {
    fn save_state(&self) -> Result<(), String> {
        self.save_state()
    }

    fn load_state(&self) -> Result<(), String> {
        self.load_state()
    }
}

#[cfg(not(feature = "durability"))]
pub trait DurableEmbed {}

#[cfg(not(feature = "durability"))]
pub trait ExtendedGuest: Guest {}

#[cfg(not(feature = "durability"))]
impl<T: Guest> ExtendedGuest for T {}