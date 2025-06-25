use base64::{engine::general_purpose, Engine as _};
use chrono::{Datelike, Timelike};
use golem_graph::golem::graph::{
    errors::GraphError,
    types::{Date, Datetime, Linestring, Point, Polygon, PropertyMap, PropertyValue, Time},
};
use serde_json::{json, Map, Value};

pub(crate) fn to_arango_value(value: PropertyValue) -> Result<Value, GraphError> {
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
        PropertyValue::Time(t) => Value::String(format!(
            "{:02}:{:02}:{:02}.{:09}",
            t.hour, t.minute, t.second, t.nanosecond
        )),
        PropertyValue::Datetime(dt) => {
            let date_str = format!(
                "{:04}-{:02}-{:02}",
                dt.date.year, dt.date.month, dt.date.day
            );
            let time_str = format!(
                "{:02}:{:02}:{:02}.{:09}",
                dt.time.hour, dt.time.minute, dt.time.second, dt.time.nanosecond
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
        PropertyValue::Duration(d) => {
            Value::String(format!("P{}S", d.seconds)) // Simplified ISO 8601 for duration
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
            json!({ "type": "LineString", "coordinates": coords })
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
            json!({ "type": "Polygon", "coordinates": rings })
        }
    })
}

pub(crate) fn to_arango_properties(
    properties: PropertyMap,
) -> Result<Map<String, Value>, GraphError> {
    let mut map = Map::new();
    for (key, value) in properties {
        map.insert(key, to_arango_value(value)?);
    }
    Ok(map)
}

pub(crate) fn from_arango_properties(
    properties: Map<String, Value>,
) -> Result<PropertyMap, GraphError> {
    let mut prop_map = Vec::new();
    for (key, value) in properties {
        prop_map.push((key, from_arango_value(value)?));
    }
    Ok(prop_map)
}

pub(crate) fn from_arango_value(value: Value) -> Result<PropertyValue, GraphError> {
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
                    "Unsupported number type from ArangoDB".to_string(),
                ))
            }
        }
        Value::String(s) => {
            if s.len() >= 4
                && s.len() % 4 == 0
                && s.chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '/' || c == '=')
            {
                if let Ok(bytes) = general_purpose::STANDARD.decode(&s) {
                    // Only treating as base64 bytes in these cases:
                    // 1. String contains base64 padding or special characters
                    // 2. String is relatively long (likely encoded data)
                    // 3. String starts with common base64 prefixes or patterns
                    if s.contains('=')
                        || s.contains('+')
                        || s.contains('/')
                        || s.len() >= 12
                        || (s.len() == 4 && bytes.len() == 3 && bytes.iter().all(|&b| b < 32))
                    {
                        return Ok(PropertyValue::Bytes(bytes));
                    }
                }
            }

            if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(&s) {
                return Ok(PropertyValue::Datetime(Datetime {
                    date: Date {
                        year: dt.year() as u32,
                        month: dt.month() as u8,
                        day: dt.day() as u8,
                    },
                    time: Time {
                        hour: dt.hour() as u8,
                        minute: dt.minute() as u8,
                        second: dt.second() as u8,
                        nanosecond: dt.nanosecond(),
                    },
                    timezone_offset_minutes: (dt.offset().local_minus_utc() / 60).try_into().ok(),
                }));
            }

            Ok(PropertyValue::StringValue(s))
        }
        Value::Object(map) => {
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
            Err(GraphError::InvalidPropertyType(
                "Unsupported object type from ArangoDB".to_string(),
            ))
        }
        Value::Array(_) => Err(GraphError::InvalidPropertyType(
            "Array properties are not directly supported, use a nested object".to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use super::*;
    use base64::engine::general_purpose;
    use chrono::{FixedOffset, TimeZone};
    use golem_graph::golem::graph::{
        errors::GraphError,
        types::{Date, Datetime, Linestring, Point, Polygon, PropertyValue, Time},
    };
    use serde_json::{json, Value};
    #[test]
    fn test_to_arango_value_primitives() {
        assert_eq!(
            to_arango_value(PropertyValue::NullValue).unwrap(),
            Value::Null
        );
        assert_eq!(
            to_arango_value(PropertyValue::Boolean(true)).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            to_arango_value(PropertyValue::Int32(42)).unwrap(),
            json!(42)
        );
        assert_eq!(
            to_arango_value(PropertyValue::Float32Value(PI)).unwrap(),
            PI
        );
        assert_eq!(
            to_arango_value(PropertyValue::StringValue("foo".into())).unwrap(),
            Value::String("foo".into())
        );
    }

    #[test]
    fn test_to_arango_value_bytes_and_date_time() {
        let data = vec![1u8, 2, 3];
        let encoded = general_purpose::STANDARD.encode(&data);
        assert_eq!(
            to_arango_value(PropertyValue::Bytes(data.clone())).unwrap(),
            Value::String(encoded)
        );

        let date = Date {
            year: 2021,
            month: 12,
            day: 31,
        };
        assert_eq!(
            to_arango_value(PropertyValue::Date(date)).unwrap(),
            Value::String("2021-12-31".into())
        );

        let time = Time {
            hour: 1,
            minute: 2,
            second: 3,
            nanosecond: 4,
        };
        assert_eq!(
            to_arango_value(PropertyValue::Time(time)).unwrap(),
            Value::String("01:02:03.000000004".into())
        );

        let dt = Datetime {
            date: Date {
                year: 2022,
                month: 1,
                day: 2,
            },
            time: Time {
                hour: 3,
                minute: 4,
                second: 5,
                nanosecond: 6,
            },
            timezone_offset_minutes: Some(0),
        };
        assert_eq!(
            to_arango_value(PropertyValue::Datetime(dt)).unwrap(),
            Value::String("2022-01-02T03:04:05.000000006Z".into())
        );
    }

    #[test]
    fn test_to_arango_value_geometries() {
        let p = Point {
            longitude: 10.0,
            latitude: 20.0,
            altitude: None,
        };
        let v = to_arango_value(PropertyValue::Point(p)).unwrap();
        assert_eq!(v, json!({"type":"Point","coordinates":[10.0,20.0]}));

        let ls = Linestring {
            coordinates: vec![
                Point {
                    longitude: 1.0,
                    latitude: 2.0,
                    altitude: Some(3.0),
                },
                Point {
                    longitude: 4.0,
                    latitude: 5.0,
                    altitude: None,
                },
            ],
        };
        let v = to_arango_value(PropertyValue::Linestring(ls)).unwrap();
        assert_eq!(
            v,
            json!({"type":"LineString","coordinates":[[1.0,2.0,3.0],[4.0,5.0]]})
        );

        let poly = Polygon {
            exterior: vec![
                Point {
                    longitude: 0.0,
                    latitude: 0.0,
                    altitude: None,
                },
                Point {
                    longitude: 1.0,
                    latitude: 0.0,
                    altitude: None,
                },
                Point {
                    longitude: 1.0,
                    latitude: 1.0,
                    altitude: None,
                },
            ],
            holes: Some(vec![vec![
                Point {
                    longitude: 0.2,
                    latitude: 0.2,
                    altitude: None,
                },
                Point {
                    longitude: 0.8,
                    latitude: 0.2,
                    altitude: None,
                },
                Point {
                    longitude: 0.8,
                    latitude: 0.8,
                    altitude: None,
                },
            ]]),
        };
        let v = to_arango_value(PropertyValue::Polygon(poly)).unwrap();
        assert!(v.get("type").unwrap() == "Polygon");
    }

    #[test]
    fn test_to_arango_properties_and_roundtrip() {
        let props = vec![
            ("a".into(), PropertyValue::Int64(7)),
            ("b".into(), PropertyValue::StringValue("x".into())),
        ];
        let map = to_arango_properties(props.clone()).unwrap();
        let round = from_arango_properties(map).unwrap();
        assert_eq!(round, props);
    }

    #[test]
    fn test_from_arango_value_base64_and_datetime() {
        let bytes = vec![9u8, 8, 7];
        let s = general_purpose::STANDARD.encode(&bytes);
        let v = from_arango_value(Value::String(s.clone())).unwrap();
        assert_eq!(v, PropertyValue::Bytes(bytes));

        let tz = FixedOffset::east_opt(0).unwrap();
        let cds = tz
            .with_ymd_and_hms(2023, 3, 4, 5, 6, 7)
            .unwrap()
            .with_nanosecond(8)
            .unwrap();
        let s2 = cds.to_rfc3339();
        let v2 = from_arango_value(Value::String(s2.clone())).unwrap();
        if let PropertyValue::Datetime(dt) = v2 {
            assert_eq!(dt.date.year, 2023);
            assert_eq!(dt.time.nanosecond, 8);
        } else {
            panic!("Expected Datetime");
        }
    }

    #[test]
    fn test_from_arango_value_geometries() {
        let p_json = json!({"type":"Point","coordinates":[1.1,2.2,3.3]});
        if let PropertyValue::Point(p) = from_arango_value(p_json).unwrap() {
            assert_eq!(p.altitude, Some(3.3));
        } else {
            panic!()
        }

        let ls_json = json!({"type":"LineString","coordinates":[[1,2],[3,4,5]]});
        if let PropertyValue::Linestring(ls) = from_arango_value(ls_json).unwrap() {
            assert_eq!(ls.coordinates.len(), 2);
        } else {
            panic!()
        }

        let poly_json = json!({"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1]],[[0.2,0.2],[0.3,0.3],[0.4,0.4]]]});
        if let PropertyValue::Polygon(poly) = from_arango_value(poly_json).unwrap() {
            assert!(poly.holes.is_some());
        } else {
            panic!()
        }
    }

    #[test]
    fn test_from_arango_value_invalid() {
        let arr = Value::Array(vec![json!(1)]);
        assert!(matches!(
            from_arango_value(arr).unwrap_err(),
            GraphError::InvalidPropertyType(_)
        ));
    }
}
