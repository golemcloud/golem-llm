use wit_bindgen::generate;

generate!({
    path: "../wit/golem-vector.wit",
    world: "vector-library",
    generate_all,
    generate_unused_types:true,
    pub_export_macro: true
});

pub use __export_vector_library_impl as export_vector;