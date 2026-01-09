{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    systems.url = "github:nix-systems/default";
  };

  outputs =
    { systems, nixpkgs, ... }:
    let
      eachSystem =
        f: nixpkgs.lib.genAttrs (import systems) (system: f (import nixpkgs { inherit system; }));
    in
    {
      devShells = eachSystem (pkgs: {
        default = pkgs.mkShell {
          packages = with pkgs; [
            python313
            python313Packages.numpy
            python313Packages.scipy
            python313Packages.networkx
            python313Packages.matplotlib
            python313Packages.opencv4
            uv
          ];
        };
      });
    };
}
