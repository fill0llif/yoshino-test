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
void hebbNetShouldLearnXor() {
	value x1AndNotX2Net = LayeredNeuralNetwork([
		LayerSetup(
			(Integer index) => "x``index``",
			(Integer index) => BipolarStep(),
			2,
			true
		),
		LayerSetup(
			(Integer index) => "y1",
			(Integer index) => BipolarStep()
		)
	]);
	value notX1AndX2Net = LayeredNeuralNetwork([
		LayerSetup(
			(Integer index) => "x``index``",
			(Integer index) => BipolarStep(),
			2,
			true
		),
		LayerSetup(
			(Integer index) => "y2",
			(Integer index) => BipolarStep()
		)
	]);
	value orNet = LayeredNeuralNetwork([
		LayerSetup(
			(Integer index) => "y``index``",
			(Integer index) => BipolarStep(),
			2,
			true
		),
		LayerSetup(
			(Integer index) => "z``index``",
			(Integer index) => BipolarStep()
		)
	]);

	{
		[x1AndNotX2Net,(Integer index) => "x``index``"] -> "bipolar_and_not.csv",
		[notX1AndX2Net,(Integer index) => "x``index``"] -> "bipolar_not_and.csv",
		[orNet,(Integer index) => "y``index``"] -> "bipolar_or.csv"
	}.each(([LayeredNeuralNetwork net,String(Integer) label] -> String filename) {
		assert (is Resource resource = `module`.resourceByPath(filename));
		value trainingInputReader = () =>
			BufferedReader(InputStreamReader(CeylonResourceInputStream(resource)));
		HebbTraining(
			net, 
			trainingInputReader)
		.train();
		assertNetLearnsOperation(net, trainingInputReader);
	});

	value xorNet = 
		let (merged = x1AndNotX2Net.Join(notX1AndX2Net).on(1, 1))
		merged.append(orNet);
	assert (is Resource resource = `module`.resourceByPath("bipolar_xor.csv"));
	value trainingInputReader = () =>
		BufferedReader(InputStreamReader(CeylonResourceInputStream(resource)));
	assertNetLearnsOperation(xorNet, trainingInputReader);
}
