import ceylon.test {
	test
}

import it.feelburst.yoshino.io {
	ResourceInputReader,
	LiveFloatValueInputReader
}
import it.feelburst.yoshino.learning.training {
	HebbTraining
}
import it.feelburst.yoshino.model {
	BipolarStep,
	Layer,
	LayeredNeuralNetwork
}

test
void hebbNetShouldLearnOr() {
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
	assert (is Resource trainingFile = `module`.resourceByPath("bipolar_or.csv"));
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
