# Third-party components and weights

This directory's orchestration code follows the license of the parent project.
Third-party pose-estimation repositories, model weights, datasets, and robot
SDKs are not relicensed by this project.

In particular, the local HumanEgo copy used during development is licensed
under **PolyForm Noncommercial 1.0.0**. Do not vendor that source tree or its
weights into an LGPL release without preserving its license and noncommercial
restriction. Configure HumanEgo/HaMeR as an optional external backend instead.

Before publishing a release:

1. List every external repository and exact revision.
2. Link its license and model-weight terms separately.
3. Do not upload third-party weights unless redistribution is explicitly
   permitted.
4. Remove private absolute paths, server addresses, serial numbers, credentials,
   and raw data that lacks a publication consent/license.
5. State clearly which parts are usable commercially and which are not.

This file is an engineering checklist, not legal advice.
