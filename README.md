# CapsNet

A development repo for testing out capsule networks. Planning on structuring
capsule network objects into an api for ease of use once it is in a more complete
state

## Contents

- Layers
    - Contains keras layers for building capsule networks
- Models
    - Contains capsule networks from the original papers as well as some experimental models
- Routing
    - Contains various routing algorithms for between capsule layers
- Lossess
    - Loss functions used in some of the capsule networks
- Tools
    - Helper functions used throughout the above scripts

## Notes

- Dynamic routing does not work with conv-caps layers
    - Poses vanish/approag pch zero