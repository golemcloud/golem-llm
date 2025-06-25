use base64::{engine::general_purpose, Engine as _};
use golem_graph::golem::graph::{
    errors::GraphError,
    types::{
        Date, Datetime, ElementId, Linestring, Point, Polygon, PropertyMap, PropertyValue, Time,
    },
};
use serde_json::{json, Map, Value};

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
        PropertyValue::Float32Value(f32) => json!(f32),
        PropertyValue::Float64Value(f64) => json!(f64),
        PropertyValue::StringValue(s) => Value::String(s),
        PropertyValue::Bytes(b) => Value::String(format!(
            "__bytes_b64__:{}",
            general_purpose::STANDARD.encode(b)
        )),
        PropertyValue::Date(d) => {
            Value::String(format!("{:04}-{:02}-{:02}", d.year, d.month, d.day))
        }
        PropertyValue::Time(t) => Value::String(format!(
            "{:02}:{:02}:{:02}.{}",
            t.hour,
            t.minute,
            t.second,
            format_args!("{:09}", t.nanosecond)
        )),
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
            let tz_str = match dt.timezone_offset_minutes {
                Some(offset) => {
                    if offset == 0 {
                        "Z".to_string()
                    } else {
                        let sign = if offset > 0 { '+' } else { '-' };
                        let hours = (offset.abs() / 60) as u8;
                        let minutes = (offset.abs() % 60) as u8;
                        format!("{}{:02}:{:02}", sign, hours, minutes)
                    }
                }
                None => "".to_string(),
            };
            Value::String(format!("{}T{}{}", date_str, time_str, tz_str))
        }
        PropertyValue::Duration(_) => {
            return Err(GraphError::UnsupportedOperation(
                "Duration conversion to JSON is not supported by Neo4j's HTTP API in this format."
                    .to_string(),
            ))
        }
        PropertyValue::Point(p) => json!({
            "type": "Point",
            "coordinates": if let Some(alt) = p.altitude {
                vec![p.longitude, p.latitude, alt]
            } else {
                vec![p.longitude, p.latitude]
            }
        }),
        PropertyValue::Linestring(ls) => {
            let coords: Vec<Vec<f64>> = ls
                .coordinates
                .into_iter()
                .map(|p| {
                    if let Some(alt) = p.altitude {
                        vec![p.longitude, p.latitude, alt]
                    } else {
                        vec![p.longitude, p.latitude]
                    }
                })
                .collect();
            json!({
                "type": "LineString",
                "coordinates": coords
            })
        }
        PropertyValue::Polygon(poly) => {
            let exterior: Vec<Vec<f64>> = poly
                .exterior
                .into_iter()
                .map(|p| {
                    if let Some(alt) = p.altitude {
                        vec![p.longitude, p.latitude, alt]
                    } else {
                        vec![p.longitude, p.latitude]
                    }
                })
                .collect();

            let mut rings = vec![exterior];

            if let Some(holes) = poly.holes {
                for hole in holes {
                    let hole_coords: Vec<Vec<f64>> = hole
                        .into_iter()
                        .map(|p| {
                            if let Some(alt) = p.altitude {
                                vec![p.longitude, p.latitude, alt]
                            } else {
                                vec![p.longitude, p.latitude]
                            }
                        })
                        .collect();
                    rings.push(hole_coords);
                }
            }
            json!({
                "type": "Polygon",
                "coordinates": rings
            })
        }
    })
}

pub(crate) fn to_cypher_properties(
    properties: PropertyMap,
) -> Result<Map<String, Value>, GraphError> {
    let mut map = Map::new();
    for (key, value) in properties {
        map.insert(key, to_json_value(value)?);
    }
    Ok(map)
}

pub(crate) fn from_cypher_element_id(value: &Value) -> Result<ElementId, GraphError> {
    if let Some(id) = value.as_i64() {
        Ok(ElementId::Int64(id))
    } else if let Some(id) = value.as_str() {
        Ok(ElementId::StringValue(id.to_string()))
    } else {
        Err(GraphError::InvalidPropertyType(
            "Unsupported element ID type from Neo4j".to_string(),
        ))
    }
}

pub(crate) fn from_cypher_properties(
    properties: Map<String, Value>,
) -> Result<PropertyMap, GraphError> {
    let mut prop_map = Vec::new();
    for (key, value) in properties {
        prop_map.push((key, from_json_value(value)?));
    }
    Ok(prop_map)
}

pub(crate) fn from_json_value(value: Value) -> Result<PropertyValue, GraphError> {
    match value {
        Value::Null => Ok(PropertyValue::NullValue),
        Value::Bool(b) => Ok(PropertyValue::Boolean(b)),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(PropertyValue::Int64(i))
            } else if let Some(f) = n.as_f64() {
                Ok(PropertyValue::Float64Value(f))
            } else {
                Err(GraphError::InvalidPropertyType(
                    "Unsupported number type from Neo4j".to_string(),
                ))
            }
        }
        Value::String(s) => {
            if let Some(b64_data) = s.strip_prefix("__bytes_b64__:") {
                return general_purpose::STANDARD
                    .decode(b64_data)
                    .map(PropertyValue::Bytes)
                    .map_err(|e| {
                        GraphError::InternalError(format!("Failed to decode base64 bytes: {}", e))
                    });
            }

            if let Ok(dt) = parse_iso_datetime(&s) {
                return Ok(PropertyValue::Datetime(dt));
            }
            if let Ok(d) = parse_iso_date(&s) {
                return Ok(PropertyValue::Date(d));
            }
            if let Ok(t) = parse_iso_time(&s) {
                return Ok(PropertyValue::Time(t));
            }

            Ok(PropertyValue::StringValue(s))
        }
        Value::Object(map) => {
            // First, try to parse as GeoJSON if it has the right structure
            if let Some(typ) = map.get("type").and_then(Value::as_str) {
                if let Some(coords_val) = map.get("coordinates") {
                    match typ {
                        "Point" => {
                            if let Ok(coords) =
                                serde_json::from_value::<Vec<f64>>(coords_val.clone())
                            {
                                if coords.len() >= 2 {
                                    return Ok(PropertyValue::Point(Point {
                                        longitude: coords[0],
                                        latitude: coords[1],
                                        altitude: coords.get(2).copied(),
                                    }));
                                }
                            }
                        }
                        "LineString" => {
                            if let Ok(coords) =
                                serde_json::from_value::<Vec<Vec<f64>>>(coords_val.clone())
                            {
                                let points = coords
                                    .into_iter()
                                    .map(|p| Point {
                                        longitude: p.first().copied().unwrap_or(0.0),
                                        latitude: p.get(1).copied().unwrap_or(0.0),
                                        altitude: p.get(2).copied(),
                                    })
                                    .collect();
                                return Ok(PropertyValue::Linestring(Linestring {
                                    coordinates: points,
                                }));
                            }
                        }
                        "Polygon" => {
                            if let Ok(rings) =
                                serde_json::from_value::<Vec<Vec<Vec<f64>>>>(coords_val.clone())
                            {
                                if let Some(exterior_coords) = rings.first() {
                                    let exterior = exterior_coords
                                        .iter()
                                        .map(|p| Point {
                                            longitude: p.first().copied().unwrap_or(0.0),
                                            latitude: p.get(1).copied().unwrap_or(0.0),
                                            altitude: p.get(2).copied(),
                                        })
                                        .collect();

                                    let holes = if rings.len() > 1 {
                                        Some(
                                            rings[1..]
                                                .iter()
                                                .map(|hole_coords| {
                                                    hole_coords
                                                        .iter()
                                                        .map(|p| Point {
                                                            longitude: p
                                                                .first()
                                                                .copied()
                                                                .unwrap_or(0.0),
                                                            latitude: p
                                                                .get(1)
                                                                .copied()
                                                                .unwrap_or(0.0),
                                                            altitude: p.get(2).copied(),
                                                        })
                                                        .collect()
                                                })
                                                .collect(),
                                        )
                                    } else {
                                        None
                                    };

                                    return Ok(PropertyValue::Polygon(Polygon { exterior, holes }));
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }

            // This handles cases where Neo4j returns complex objects that aren't GeoJSON
            Ok(PropertyValue::StringValue(
                serde_json::to_string(&Value::Object(map)).unwrap_or_else(|_| "{}".to_string()),
            ))
        }
        _ => Err(GraphError::InvalidPropertyType(
            "Unsupported property type from Neo4j".to_string(),
        )),
    }
}

fn parse_iso_date(s: &str) -> Result<Date, ()> {
    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() != 3 {
        return Err(());
    }
    let year = parts[0].parse().map_err(|_| ())?;
    let month = parts[1].parse().map_err(|_| ())?;
    let day = parts[2].parse().map_err(|_| ())?;
    Ok(Date { year, month, day })
}

fn parse_iso_time(s: &str) -> Result<Time, ()> {
    let time_part = s
        .split_once('Z')
        .or_else(|| s.split_once('+'))
        .or_else(|| s.split_once('-'))
        .map_or(s, |(tp, _)| tp);
    let main_parts: Vec<&str> = time_part.split(':').collect();
    if main_parts.len() != 3 {
        return Err(());
    }
    let hour = main_parts[0].parse().map_err(|_| ())?;
    let minute = main_parts[1].parse().map_err(|_| ())?;
    let (second, nanosecond) = if main_parts[2].contains('.') {
        let sec_parts: Vec<&str> = main_parts[2].split('.').collect();
        let s = sec_parts[0].parse().map_err(|_| ())?;
        let ns_str = format!("{:0<9}", sec_parts[1]);
        let ns = ns_str[..9].parse().map_err(|_| ())?;
        (s, ns)
    } else {
        (main_parts[2].parse().map_err(|_| ())?, 0)
    };

    Ok(Time {
        hour,
        minute,
        second,
        nanosecond,
    })
}

fn parse_iso_datetime(s: &str) -> Result<Datetime, ()> {
    let (date_str, time_str) = s.split_once('T').ok_or(())?;
    let date = parse_iso_date(date_str)?;
    let time = parse_iso_time(time_str)?;

    let timezone_offset_minutes = if time_str.ends_with('Z') {
        Some(0)
    } else if let Some((_, tz)) = time_str.rsplit_once('+') {
        let parts: Vec<&str> = tz.split(':').collect();
        if parts.len() != 2 {
            return Err(());
        }
        let hours: i16 = parts[0].parse().map_err(|_| ())?;
        let minutes: i16 = parts[1].parse().map_err(|_| ())?;
        Some(hours * 60 + minutes)
    } else if let Some((_, tz)) = time_str.rsplit_once('-') {
        let parts: Vec<&str> = tz.split(':').collect();
        if parts.len() != 2 {
            return Err(());
        }
        let hours: i16 = parts[0].parse().map_err(|_| ())?;
        let minutes: i16 = parts[1].parse().map_err(|_| ())?;
        Some(-(hours * 60 + minutes))
    } else {
        None
    };

    Ok(Datetime {
        date,
        time,
        timezone_offset_minutes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use golem_graph::golem::graph::types::{Date, Datetime, Point, Time};

    #[test]
    fn test_simple_values_roundtrip() {
        let original = PropertyValue::Int64(12345);
        let json_val = to_json_value(original.clone()).unwrap();
        let converted = from_json_value(json_val).unwrap();

        match (original, converted) {
            (PropertyValue::Int64(o), PropertyValue::Int64(c)) => assert_eq!(o, c),
            (o, c) => panic!("Type mismatch: expected {:?} got {:?}", o, c),
        }
    }

    #[test]
    fn test_datetime_values_roundtrip() {
        let original = PropertyValue::Datetime(Datetime {
            date: Date {
                year: 2024,
                month: 7,
                day: 18,
            },
            time: Time {
                hour: 10,
                minute: 30,
                second: 0,
                nanosecond: 123456789,
            },
            timezone_offset_minutes: Some(120),
        });

        let json_val = to_json_value(original.clone()).unwrap();
        let converted = from_json_value(json_val).unwrap();

        match (original, converted) {
            (PropertyValue::Datetime(o), PropertyValue::Datetime(c)) => {
                assert_eq!(o.date.year, c.date.year);
                assert_eq!(o.date.month, c.date.month);
                assert_eq!(o.date.day, c.date.day);
                assert_eq!(o.time.hour, c.time.hour);
                assert_eq!(o.time.minute, c.time.minute);
                assert_eq!(o.time.second, c.time.second);
                assert_eq!(o.time.nanosecond, c.time.nanosecond);
                assert_eq!(o.timezone_offset_minutes, c.timezone_offset_minutes);
            }
            (o, c) => panic!("Type mismatch: expected {:?} got {:?}", o, c),
        }
    }

    #[test]
    fn test_point_values_roundtrip() {
        let original = PropertyValue::Point(Point {
            longitude: 1.23,
            latitude: 4.56,
            altitude: Some(7.89),
        });

        let json_val = to_json_value(original.clone()).unwrap();
        let converted = from_json_value(json_val).unwrap();

        match (original, converted) {
            (PropertyValue::Point(o), PropertyValue::Point(c)) => {
                assert!((o.longitude - c.longitude).abs() < f64::EPSILON);
                assert!((o.latitude - c.latitude).abs() < f64::EPSILON);
                assert_eq!(o.altitude.is_some(), c.altitude.is_some());
                if let (Some(o_alt), Some(c_alt)) = (o.altitude, c.altitude) {
                    assert!((o_alt - c_alt).abs() < f64::EPSILON);
                }
            }
            (o, c) => panic!("Type mismatch: expected {:?} got {:?}", o, c),
        }
    }

    #[test]
    fn test_unsupported_duration_conversion() {
        let original = PropertyValue::Duration(golem_graph::golem::graph::types::Duration {
            seconds: 10,
            nanoseconds: 0,
        });

        let result = to_json_value(original);
        assert!(matches!(result, Err(GraphError::UnsupportedOperation(_))));
    }
}
