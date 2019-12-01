import ceylon.test {
	assertEquals,
	assertTrue
}

import it.feelburst.yoshino.io {
	SupervisedPairs
}
import it.feelburst.yoshino.model {
	LayeredNeuralNetwork,
	Function
}

import java.io {
	BufferedReader
}
shared void assertNetLearnsOperation(
	LayeredNeuralNetwork neuralNetwork, 
	BufferedReader trainingInputReader()) =>
	//for each training pair (operation configuration)
	SupervisedPairs(
		trainingInputReader,
		neuralNetwork.inputs(false).size)
	.each(({Float*} trainings -> {Float*} targets) {
		//assert if computed outputs are equal to the given targets
		value outputs = neuralNetwork.apply(trainings);
		assertEquals(outputs.sequence(), targets.sequence());
	});

shared void assertNetLearnOperationCombinationWithinError(
	LayeredNeuralNetwork neuralNetwork, 
	BufferedReader trainingInputReader(),
	Float error,
	Function loss(Float target)) {
	value size = SupervisedPairs(
		trainingInputReader,
		neuralNetwork.inputs(false).size).size;
	assertTrue(size > 0);
	value averageLoss =
	SupervisedPairs(
		trainingInputReader,
		neuralNetwork.inputs(false).size)
	.map(({Float*} trainings -> {Float*} targets) =>
		let (outputs = neuralNetwork.apply(trainings))
		zipEntries(outputs,targets)
		.map((Float output -> Float target) =>
			loss(target).apply(output))
		.fold(0.0)((Float partial, Float element) =>
			partial + element))
	.fold(0.0)((Float partial, Float element) =>
		partial + element) / size;
	assertTrue(averageLoss < error);
}