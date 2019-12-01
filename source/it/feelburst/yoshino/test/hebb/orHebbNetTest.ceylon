import ceylon.test {
	test
}

import it.feelburst.yoshino.io {
	CeylonResourceInputStream
}
import it.feelburst.yoshino.model {
	BipolarStep,
	LayeredNeuralNetwork,
	LayerSetup
}
import it.feelburst.yoshino.training.hebb {
	HebbTraining
}

import java.io {
	BufferedReader,
	InputStreamReader
}
import it.feelburst.yoshino.test {

	assertNetLearnsOperation
}

test
void hebbNetShouldLearnOr() {
	value net = LayeredNeuralNetwork([
		LayerSetup(
			(Integer index) => "x``index``",
			(Integer index) => BipolarStep(),
			2,
			true
		),
		LayerSetup(
			(Integer index) => "y``index``",
			(Integer index) => BipolarStep()
		)
	]);
	assert (is Resource resource = `module`.resourceByPath("bipolar_or.csv"));
	value trainingInputReader = () =>
		BufferedReader(InputStreamReader(CeylonResourceInputStream(resource)));
	HebbTraining(
		net, 
		trainingInputReader)
	.train();
	assertNetLearnsOperation(net, trainingInputReader);
}
