fn main() {
    #[cfg(feature = "coordinator")]
    {
        let proto_file = "../../proto/coordinator.proto";
        let proto_dir = "../../proto";
        let out_dir = "src/coordinator/proto";

        // Rerun if proto file changes
        println!("cargo:rerun-if-changed={}", proto_file);

        // Ensure output directory exists
        std::fs::create_dir_all(out_dir).expect("Failed to create proto output directory");

        tonic_build::configure()
            .build_server(true)
            .build_client(true)
            .type_attribute(".", "#[derive(serde::Serialize, serde::Deserialize)]")
            .out_dir(out_dir)
            .compile_protos(&[proto_file], &[proto_dir])
            .expect("Failed to compile coordinator.proto");
    }
}
