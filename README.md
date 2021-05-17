# Persuasive Development Environment

This program and the accompanying materials are made available under the terms of
the Eclipse Public License 2.0 which is available at <http://www.eclipse.org/legal/epl-2.0>.

## Scope

Urban traffic congestion is a significant problem that has repercussions for economic growth and the wellness of urban populations. Large-scale social events (e.g., sports events, concerts, or demonstrations) represent an integral aspect of living in a city but are frequently a source of severe traffic disruption, with an impact on both the people participating in the event and the wider community.

With urban transportation infrastructures fast reaching their capacities, the only viable strategy to alleviate congestion is to optimise the use of the available resources. *We hypothesise that better coordination and cooperation between travellers could help to reduce congestion, for example, by balancing the load among the different modes of transport and reducing the time required to find parking spaces*. Motivated by the availability of fine-grained sensor data from individuals and vehicles and the opportunity to influence traveller behaviour in real-time through appropriate feedback, the objective of this work is to implement dynamic multi-modal traffic congestion mitigation strategies for large-scale social events, improving travel and parking delays through load balancing of the transportation modes and staggered departure times.

Our approach is to use machine learning techniques to identify travel optimisation strategies that also leverage IoT infrastructures for participatory sensing and data dissemination. *Through reinforcement learning, we expect to recognise underlying patterns in the mobility surrounding events that can be exploited to optimise journey planning.*

Our proposed solution comprises of planning strategies that can be proposed to travellers as guidance and instructions. Considering the constraints arising from large-scale social events, *we aim to present dynamic strategies for congestion mitigation based on participatory sensing and feedback-loop information that optimise the use of multi-modal transportation capacity*.

In the scope of this research project, the only information that may be required from individuals is anonymised location. Throughout the design and implementation process, anonymisation strategies will be used to guarantee that no identifiable personal data will be collected or processed by the system.

**_This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No. 713567. ENABLE is funded under Science Foundation Ireland (16/SP/3804) and is co-funded under the European Regional Development Fund._**

### Contacts

- Dr. Lara CODECA <[lara.codeca@gmail.com](mailto:lara.codeca@gmail.com)>
- Prof. Vinny CAHILL <[vinny.cahill@tcd.ie](mailto:vinny.cahill@tcd.ie)>

### Site

[https://lcodeca.github.io/persuasive/](https://lcodeca.github.io/persuasive/)

## Persuasive development infrastructure

### RLlib SUMO Utils

In order to connect the SUMO simulator with the RLlib distributed environment, we implemented a [python library](https://github.com/lcodeca/rllibsumoutils) and a [Docker environment](https://github.com/lcodeca/rllibsumodocker) for testing.

**Note: the library has been included in [RLlib master](https://github.com/ray-project/ray/pull/11710) and is available from RLlib 1.0.1.**

### Random Grid Scenario

The random grid scenario used for testing is available at [https://github.com/lcodeca/random-grid-sumo](https://github.com/lcodeca/random-grid-sumo)

## Preliminary results

The preliminary results from Persuasive are available on GitHub at [https://github.com/lcodeca/results/Persuasive-training](https://github.com/lcodeca/results/Persuasive-training)
