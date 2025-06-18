use std::time::Duration;

/// An Event
#[derive(Default, Debug, Eq, PartialEq, Clone)]
pub struct MessageEvent {
    /// The event name if given
    pub event: String,
    /// The event data
    pub data: String,
    /// The event id if given
    pub id: String,
    /// Retry duration if given
    pub retry: Option<Duration>,
}
