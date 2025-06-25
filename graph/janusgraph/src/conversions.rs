use base64::{engine::general_purpose, Engine as _};
use golem_graph::golem::graph::{
    errors::GraphError,
    types::{Date, Datetime, Point, PropertyValue, Time},
};
use serde_json::{json, Value};

pub(crate) fn to_json_value(value: PropertyValue) -> Result<Value, GraphError> {
    Ok(match value {
        PropertyValue::NullValue => Value::Null,
        PropertyValue::Boolean(b) => Value::Bool(b),
        PropertyValue::Int8(i) => json!(i),
        PropertyValue::Int16(i) => json!(i),
        PropertyValue::Int32(i) => json!(i),
        PropertyValue::Int64(i) => json!(i),
        PropertyValue::Uint8(i) => json!(i),
        PropertyValue::Uint16(i) => json!(i),
        PropertyValue::Uint32(i) => json!(i),
        PropertyValue::Uint64(i) => json!(i),
        PropertyValue::Float32Value(f) => json!(f),
        PropertyValue::Float64Value(f) => json!(f),
        PropertyValue::StringValue(s) => Value::String(s),
        PropertyValue::Bytes(b) => Value::String(general_purpose::STANDARD.encode(b)),
        PropertyValue::Date(d) => {
            Value::String(format!("{:04}-{:02}-{:02}", d.year, d.month, d.day))
        }
        PropertyValue::Datetime(dt) => {
            let date_str = format!(
                "{:04}-{:02}-{:02}",
                dt.date.year, dt.date.month, dt.date.day
            );
            let time_str = format!(
                "{:02}:{:02}:{:02}.{}",
                dt.time.hour,
                dt.time.minute,
                dt.time.second,
                format_args!("{:09}", dt.time.nanosecond)
            );
            Value::String(format!("{}T{}Z", date_str, time_str))
        }
        PropertyValue::Point(p) => {
            if let Some(alt) = p.altitude {
                Value::String(format!("POINT ({} {} {})", p.longitude, p.latitude, alt))
            } else {
                Value::String(format!("POINT ({} {})", p.longitude, p.latitude))
            }
        }
        _ => {
            return Err(GraphError::UnsupportedOperation(
                "This property type is not supported as a Gremlin binding".to_string(),
            ))
        }
    })
}

pub(crate) fn from_gremlin_value(value: &Value) -> Result<PropertyValue, GraphError> {
    match value {
        Value::Null => Ok(PropertyValue::NullValue),
        Value::Bool(b) => Ok(PropertyValue::Boolean(*b)),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(PropertyValue::Int64(i))
            } else if let Some(f) = n.as_f64() {
                Ok(PropertyValue::Float64Value(f))
            } else {
                Err(GraphError::InvalidPropertyType(
                    "Unsupported number type from Gremlin".to_string(),
                ))
            }
        }
        Value::String(s) => {
            if let Ok(dt) = parse_iso_datetime(s) {
                return Ok(PropertyValue::Datetime(dt));
            }
            if let Ok(d) = parse_iso_date(s) {
                return Ok(PropertyValue::Date(d));
            }
            if let Ok(p) = parse_wkt_point(s) {
                return Ok(PropertyValue::Point(p));
            }
            Ok(PropertyValue::StringValue(s.clone()))
        }
        Value::Object(obj) => {
            // Handle GraphSON wrapped values like {"@type": "g:Int64", "@value": 29}
            if let (Some(Value::String(gtype)), Some(gvalue)) =
                (obj.get("@type"), obj.get("@value"))
            {
                match gtype.as_str() {
                    "g:Int64" | "g:Int32" | "g:Int16" | "g:Int8" => {
                        if let Some(i) = gvalue.as_i64() {
                            Ok(PropertyValue::Int64(i))
                        } else {
                            Err(GraphError::InvalidPropertyType(
                                "Invalid GraphSON integer value".to_string(),
                            ))
                        }
                    }
                    "g:Float" | "g:Double" => {
                        if let Some(f) = gvalue.as_f64() {
                            Ok(PropertyValue::Float64Value(f))
                        } else {
                            Err(GraphError::InvalidPropertyType(
                                "Invalid GraphSON float value".to_string(),
                            ))
                        }
                    }
                    _ => {
                        // For other GraphSON types, try to parse the @value recursively
                        from_gremlin_value(gvalue)
                    }
                }
            } else {
                Err(GraphError::InvalidPropertyType(
                    "Gremlin objects without GraphSON @type/@value cannot be converted to a WIT property type.".to_string(),
                ))
            }
        }
        Value::Array(_) => Err(GraphError::InvalidPropertyType(
            "Gremlin arrays cannot be converted to a WIT property type.".to_string(),
        )),
    }
}

fn parse_wkt_point(s: &str) -> Result<Point, ()> {
    if !s.starts_with("POINT") {
        return Err(());
    }
    let content = s.trim_start_matches("POINT").trim();
    let content = content.strip_prefix('(').unwrap_or(content);
    let content = content.strip_suffix(')').unwrap_or(content);

    let parts: Vec<&str> = content.split_whitespace().collect();
    if parts.len() < 2 {
        return Err(());
    }

    let lon = parts[0].parse::<f64>().map_err(|_| ())?;
    let lat = parts[1].parse::<f64>().map_err(|_| ())?;
    let alt = if parts.len() > 2 {
        parts[2].parse::<f64>().ok()
    } else {
        None
    };

    Ok(Point {
        longitude: lon,
        latitude: lat,
        altitude: alt,
    })
}

fn parse_iso_date(s: &str) -> Result<Date, ()> {
    if s.len() != 10 {
        return Err(());
    }
    if s.chars().nth(4) != Some('-') || s.chars().nth(7) != Some('-') {
        return Err(());
    }

    let year = s[0..4].parse().map_err(|_| ())?;
    let month = s[5..7].parse().map_err(|_| ())?;
    let day = s[8..10].parse().map_err(|_| ())?;

    Ok(Date { year, month, day })
}

fn parse_iso_datetime(s: &str) -> Result<Datetime, ()> {
    if s.len() < 19 {
        return Err(());
    }
    let date_part = &s[0..10];
    let time_part = &s[11..];

    let date = parse_iso_date(date_part)?;

    let hour = time_part[0..2].parse().map_err(|_| ())?;
    let minute = time_part[3..5].parse().map_err(|_| ())?;
    let second = time_part[6..8].parse().map_err(|_| ())?;

    let nanosecond = if time_part.len() > 9 && time_part.chars().nth(8) == Some('.') {
        let nano_str = &time_part[9..];
        let nano_str_padded = format!("{:0<9}", nano_str);
        nano_str_padded[0..9].parse().map_err(|_| ())?
    } else {
        0
    };

    Ok(Datetime {
        date,
        time: Time {
            hour,
            minute,
            second,
            nanosecond,
        },
        timezone_offset_minutes: Some(0), // Gremlin dates are timezone-aware
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use golem_graph::golem::graph::types::{Duration, PropertyValue};

    #[test]
    fn test_unsupported_duration_conversion() {
        let original = PropertyValue::Duration(Duration {
            seconds: 10,
            nanoseconds: 0,
        });

        let result = to_json_value(original);
        assert!(matches!(result, Err(GraphError::UnsupportedOperation(_))));
    }
}
