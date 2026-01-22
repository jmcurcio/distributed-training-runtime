fn main() {
    let proto_file = "../../proto/coordinator.proto";
    let proto_dir = "../../proto";

    // Rerun if proto file changes
    println!("cargo:rerun-if-changed={}", proto_file);

    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .out_dir("src/proto")
        .compile_protos(&[proto_file], &[proto_dir])
        .expect("Failed to compile coordinator.proto");
}
