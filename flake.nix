{
  description = "CheXpert uncertainty-label policies: dev shell for the notebook and tooling";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-darwin"
      ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = [
            # Python 3.13 to match Kaggle's image (Colab runtime base). CPU-only torch: local work
            # runs on the fake dataset; real training happens on Kaggle.
            (pkgs.python313.withPackages (
              ps: with ps; [
                torch
                torchvision
                numpy
                pandas
                scipy
                scikit-learn
                matplotlib
                opencv4
                pillow
                jupytext
                nbconvert
                ipykernel
              ]
            ))
            pkgs.ruff
            pkgs.pre-commit
            pkgs.gnumake
            pkgs.git
          ];
        };
      });
    };
}
