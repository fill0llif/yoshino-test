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
	LayeredNeuralNetwork,
	NeuralNetworkMerger,
	Neuron
}
test
void xorPerceptronOutputShouldEqualTarget() {
	value random = DefaultRandom();
	value x1AndNotX2Threshold = random.nextFloat();
	value notX1AndX2Threshold = random.nextFloat();
	value orThreshold = random.nextFloat();
	
	value x1AndNotX2Net = LayeredNeuralNetwork.byLayers([
		Layer(
			(Integer index) => "x``index``",
			(Integer index) => BipolarStep(),
			2,
			true
		),
		Layer(
			(Integer index) => "y1",
			(Integer index) => UndecidedBand(x1AndNotX2Threshold)
		)
	]);
	
	value orNet = LayeredNeuralNetwork.byLayers([
		Layer(
			(Integer index) => "y``index``",
			(Integer index) => BipolarStep(),
			2,
			true
		),
		Layer(
			(Integer index) => "z``index``",
			(Integer index) => UndecidedBand(orThreshold)
		)
	]);
	
	value notX1AndX2Net = LayeredNeuralNetwork.byLayers([
		Layer(
			(Integer index) => "x``index``",
			(Integer index) => BipolarStep(),
			2,
			true
		),
		Layer(
			(Integer index) => "y2",
			(Integer index) => UndecidedBand(notX1AndX2Threshold)
		)
	]);
	
	{
		[x1AndNotX2Net,x1AndNotX2Threshold,(Integer index) => "x``index``"] -> "bipolar_and_not.csv",
		[notX1AndX2Net,notX1AndX2Threshold,(Integer index) => "x``index``"] -> "bipolar_not_and.csv",
		[orNet,orThreshold,(Integer index) => "y``index``"] -> "bipolar_or.csv"
	}.each(([LayeredNeuralNetwork net,Float threshold,String(Integer) label] -> String filename) {
		assert (is Resource trainingFile = `module`.resourceByPath(filename));
		value trainingInputReader = RollingCachedFloatValueInputReader(ResourceInputReader(trainingFile));
		PerceptronTraining(net, nextExclusiveFloat(random), trainingInputReader).train();
		assertNetLearnOperationCombinations(net, trainingInputReader);
		assertResponseLieInCorrectSemiplane(
			net, 
			label,
			(Float output) =>
				(Float response) =>
					if (output.positive) then response > threshold
					else response < - threshold, 
			trainingInputReader);
	});
	
	value constructor = `new LayeredNeuralNetwork.byNeurons`
			.apply<LayeredNeuralNetwork,[[[Neuron+]+]]>();
	value merger = NeuralNetworkMerger(constructor);
	value xorNet = 
		let (merged = merger.merge(x1AndNotX2Net, notX1AndX2Net))
		merger.append(merged, orNet);
	assert (is Resource trainingFile = `module`.resourceByPath("bipolar_xor.csv"));
	value trainingInputReader = RollingCachedFloatValueInputReader(ResourceInputReader(trainingFile));
	assertNetLearnOperationCombinations(xorNet, trainingInputReader);
}