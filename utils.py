from diart import SpeakerDiarization, SpeakerDiarizationConfig
from diart.inference import Benchmark
from diart.models import EmbeddingModel, SegmentationModel


def benchmark(embedding, segmentation):
    embedding = EmbeddingModel.from_pretrained(embedding)
    segmentation = SegmentationModel.from_pretrained(segmentation)
    config = SpeakerDiarizationConfig(embedding=embedding, segmentation=segmentation)
    benchmark = Benchmark("ami/wav", "ami/rttm")
    return benchmark(SpeakerDiarization, config)
