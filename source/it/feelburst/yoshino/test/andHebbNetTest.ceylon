import ceylon.test {
	test
}

import it.feelburst.yoshino.io {
	LiveFloatValueInputReader,
	ResourceInputReader
}
import it.feelburst.yoshino.learning.training {
	HebbTraining
}
import it.feelburst.yoshino.model {
	BipolarStep,
	LayeredNeuralNetwork,
	Layer
}

test
void hebbNetShouldLearnAnd() {
	value net = LayeredNeuralNetwork.byLayers([
		Layer(
			(Integer index) => "x``index``",
			(Integer index) => BipolarStep(),
			2,
			true
		),
		Layer(
			(Integer index) => "y``index``",
			(Integer index) => BipolarStep()
		)
	]);
	assert (is Resource trainingFile = `module`.resourceByPath("bipolar_and.csv"));
	value trainingInputReader = LiveFloatValueInputReader(ResourceInputReader(trainingFile));
	HebbTraining(net, trainingInputReader).train();
	assertNetLearnOperationCombinations(net, trainingInputReader);
	assertResponseLieInCorrectSemiplane(
		net, 
		(Integer index) => "x``index``",
		(Float output) =>
			(Float response) =>
				if (output.negative) then response < 0.0
				else response >= 0.0, 
		trainingInputReader);
}
