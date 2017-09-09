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
	LayeredNeuralNetwork,
	Layer,
	NeuralNetworkMerger,
	Neuron
}

test
void hebbNetShouldLearnXor() {
	value x1AndNotX2Net = LayeredNeuralNetwork.byLayers([
		Layer(
			(Integer index) => "x``index``",
			(Integer index) => BipolarStep(),
			2,
			true
		),
		Layer(
			(Integer index) => "y1",
			(Integer index) => BipolarStep()
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
			(Integer index) => BipolarStep()
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
			(Integer index) => BipolarStep()
		)
	]);

	{
		[x1AndNotX2Net,(Integer index) => "x``index``"] -> "bipolar_and_not.csv",
		[notX1AndX2Net,(Integer index) => "x``index``"] -> "bipolar_not_and.csv",
		[orNet,(Integer index) => "y``index``"] -> "bipolar_or.csv"
	}.each(([LayeredNeuralNetwork net,String(Integer) label] -> String filename) {
		assert (is Resource trainingFile = `module`.resourceByPath(filename));
		value trainingInputReader = LiveFloatValueInputReader(ResourceInputReader(trainingFile));
		HebbTraining(net, trainingInputReader).train();
		assertNetLearnOperationCombinations(net, trainingInputReader);
		assertResponseLieInCorrectSemiplane(
			net, 
			label,
			(Float output) =>
				(Float response) =>
					if (output.negative) then response < 0.0
					else response >= 0.0, 
			trainingInputReader);
	});

	value constructor = `new LayeredNeuralNetwork.byNeurons`
		.apply<LayeredNeuralNetwork,[[[Neuron+]+]]>();
	value merger = NeuralNetworkMerger(constructor);
	value xorNet = 
		let (merged = merger.merge(x1AndNotX2Net, notX1AndX2Net))
		merger.append(merged, orNet);
	assert (is Resource trainingFile = `module`.resourceByPath("bipolar_xor.csv"));
	value trainingInputReader = LiveFloatValueInputReader(ResourceInputReader(trainingFile));
	assertNetLearnOperationCombinations(xorNet, trainingInputReader);
}
