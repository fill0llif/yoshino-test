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
import it.feelburst.yoshino.training.perceptron {
	PerceptronTraining
}

import java.io {
	BufferedReader,
	InputStreamReader
}
import it.feelburst.yoshino.test {

	assertNetLearnsOperation,
	nextExclusiveFloat
}
test
void xorPerceptronOutputShouldEqualTarget() {
	value random = DefaultRandom();
	value x1AndNotX2Threshold = random.nextFloat();
	value notX1AndX2Threshold = random.nextFloat();
	value orThreshold = random.nextFloat();
	
	value x1AndNotX2Net = LayeredNeuralNetwork([
		LayerSetup(
			(Integer index) => "x``index``",
			(Integer index) => BipolarStep(),
			2,
			true
		),
		LayerSetup(
			(Integer index) => "y1",
			(Integer index) => UndecidedBand(x1AndNotX2Threshold)
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
			(Integer index) => UndecidedBand(notX1AndX2Threshold)
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
			(Integer index) => UndecidedBand(orThreshold)
		)
	]);
	
	{
		[x1AndNotX2Net,x1AndNotX2Threshold,(Integer index) => "x``index``"] -> "bipolar_and_not.csv",
		[notX1AndX2Net,notX1AndX2Threshold,(Integer index) => "x``index``"] -> "bipolar_not_and.csv",
		[orNet,orThreshold,(Integer index) => "y``index``"] -> "bipolar_or.csv"
	}.each(([LayeredNeuralNetwork net,Float threshold,String(Integer) label] -> String filename) {
		assert (is Resource resource = `module`.resourceByPath(filename));
		value trainingInputReader = () =>
			BufferedReader(InputStreamReader(CeylonResourceInputStream(resource)));
		PerceptronTraining(
			net, 
			nextExclusiveFloat(random), 
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