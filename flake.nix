{
  description = "SyntaxDot Python";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, utils }:
    utils.lib.eachSystem  [ "x86_64-linux" ] (system:
    let
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfreePredicate = pkg: builtins.elem (pkgs.lib.getName pkg) [
            "libtorch"
          ];
        };
      };
    in {
      devShell = with pkgs; mkShell {
        nativeBuildInputs = [ cmake pkg-config rustup ];

        buildInputs = [
          openssl
          (python3.withPackages (ps: with ps; [
            requests
            setuptools-rust
          ]))
        ];

        LIBTORCH = symlinkJoin {
          name = "torch-join";
          paths = [ libtorch-bin.dev libtorch-bin.out ];
        };

      };
    });
}
