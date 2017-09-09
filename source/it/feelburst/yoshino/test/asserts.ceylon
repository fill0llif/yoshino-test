import ceylon.test {
	assertEquals,
	assertTrue
}

import it.feelburst.yoshino.io {
	ValueInputReader
}
import it.feelburst.yoshino.learning.training {
	SupervisedTrainingInputReaderImpl
}
import it.feelburst.yoshino.model {
	LayeredNeuralNetwork,
	Neuron,
	Bias
}
void assertNetLearnOperationCombinations(
	LayeredNeuralNetwork neuralNetwork, 
	ValueInputReader<Float> trainingInputReader) {
	try (reader = SupervisedTrainingInputReaderImpl(trainingInputReader)) {
		while (exists trainingInputs -> trainingOutputs = 
			reader.readTrainingPair(neuralNetwork.inputs(false).size)) {
			value outputs = neuralNetwork.apply(trainingInputs);
			assertEquals (outputs, trainingOutputs);
		}
	}
}
void assertResponseLieInCorrectSemiplane(
	LayeredNeuralNetwork neuralNetwork,
	String label(Integer index),
	Boolean(Float) compareResponseTo(Float output), 
	ValueInputReader<Float> trainingInputReader) {
	
	try (reader = SupervisedTrainingInputReaderImpl(trainingInputReader)) {
		while (exists trainingInputs -> trainingOutputs = reader.readTrainingPair(neuralNetwork.inputs(false).size)) {
			assert (neuralNetwork.inputs(false).size == trainingInputs.size);
			assert (neuralNetwork.outputs.size == trainingOutputs.size);
			zipEntries(neuralNetwork.outputs,trainingOutputs)
			.each((Neuron output -> Float trainingOutput) {
				value biasAndTrainingInputs = if (is Bias bias = neuralNetwork.bias(0)) then
					[bias.input,*trainingInputs]
				else
					trainingInputs;
				value inputsWeights = biasAndTrainingInputs
				.indexed
				.collect((Integer index -> Float trainingInput) {
					assert (exists input = neuralNetwork.neuron(label(index)));
					assert (exists synapse = neuralNetwork.synapsesFromTo(input, output));
					return trainingInput -> synapse.weight;
				});
				value response = inputsWeights.fold(0.0)
					((Float partial, Float input -> Float weight) => 
						partial + input * weight);
				assertTrue(compareResponseTo(trainingOutput)(response));
			});
		}
	}
}