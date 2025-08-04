from coffea import processor
import awkward as ak


class TreeExplorer(processor.ProcessorABC):
    def process(self, events):
        dataset = (
            events.metadata["dataset"] if "dataset" in events.metadata else "unknown"
        )

        data = {branch: ak.to_list(events[branch]) for branch in events.fields}
        data["entries"] = len(events)

        return {dataset: data}

    def postprocess(self, accumulator):
        return accumulator
