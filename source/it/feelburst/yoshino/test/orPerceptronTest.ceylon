import ceylon.random {
	DefaultRandom
}
import ceylon.test {
	test
}

import it.feelburst.yoshino.io {
	ResourceInputReader,
	RollingCachedFloatValueInputReader
}
import it.feelburst.yoshino.learning.training {
	PerceptronTraining
}
import it.feelburst.yoshino.model {
	BipolarStep,
	UndecidedBand,
	Layer,
	LayeredNeuralNetwork
}
test
void perceptronShouldLearnOr() {
	value random = DefaultRandom();
	value threshold = random.nextFloat();
	value net = LayeredNeuralNetwork.byLayers([
		Layer(
			(Integer index) => "x``index``",
			(Integer index) => BipolarStep(),
			2,
			true
		),
		Layer(
			(Integer index) => "y``index``",
			(Integer index) => UndecidedBand(threshold)
		)
	]);
	assert (is Resource trainingFile = `module`.resourceByPath("bipolar_or.csv"));
	value trainingInputReader = RollingCachedFloatValueInputReader(ResourceInputReader(trainingFile));
	PerceptronTraining(net, nextExclusiveFloat(random), trainingInputReader).train();
	assertNetLearnOperationCombinations(net, trainingInputReader);
	assertResponseLieInCorrectSemiplane(
		net, 
		(Integer index) => "x``index``",
		(Float output) =>
			(Float response) =>
				if (output.positive) then response > threshold
				else response < - threshold, 
		trainingInputReader);
}
