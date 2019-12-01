import ceylon.random {
	DefaultRandom
}
import ceylon.test {
	test
}

import it.feelburst.yoshino.io {
	CeylonResourceInputStream
}
import it.feelburst.yoshino.model {
	BipolarStep,
	UndecidedBand,
	LayeredNeuralNetwork,
	LayerSetup
}
import it.feelburst.yoshino.test {
	assertNetLearnsOperation,
	nextExclusiveFloat
}
import it.feelburst.yoshino.training.perceptron {
	PerceptronTraining
}

import java.io {
	BufferedReader,
	InputStreamReader
}

test
void perceptronShouldLearnAnd() {
	value random = DefaultRandom();
	value threshold = random.nextFloat();
	value net = LayeredNeuralNetwork([
		LayerSetup(
			(Integer index) => "x``index``",
			(Integer index) => BipolarStep(),
			2,
			true
		),
		LayerSetup(
			(Integer index) => "y``index``",
			(Integer index) => UndecidedBand(threshold)
		)
	]);
	assert (is Resource resource = `module`.resourceByPath("bipolar_and.csv"));
	value trainingInputReader = () =>
		BufferedReader(InputStreamReader(CeylonResourceInputStream(resource)));
	PerceptronTraining(
		net, 
		nextExclusiveFloat(random), 
		trainingInputReader)
	.train();
	assertNetLearnsOperation(net, trainingInputReader);
}