import ceylon.test {
	test
}

import it.feelburst.yoshino.io {
	CeylonResourceInputStream
}
import it.feelburst.yoshino.learning.backpropagation {
	BackpropagationRule
}
import it.feelburst.yoshino.learning.model {
	WeightGenerator,
	NguyenWidrowWeightGenerator,
	RandomWeightGenerator
}
import it.feelburst.yoshino.model {
	LayeredNeuralNetwork,
	BipolarSigmoid,
	LayerSetup,
	identity,
	Synapse,
	Function
}
import it.feelburst.yoshino.test {
	assertNetLearnOperationCombinationWithinError
}
import it.feelburst.yoshino.training {
	SupervisedTrainingAdapter
}
import it.feelburst.yoshino.training.backpropagation {
	BackpropagationTraining,
	PatternMemorizationTrainingStrategy,
	TrainingStrategy
}

import java.io {
	BufferedReader,
	InputStreamReader
}

test
void backpropNetShouldLearnXor() {
	value error = 0.05;
	value net = LayeredNeuralNetwork([
		LayerSetup(
			(Integer index) => "x``index``",
			(Integer index) => identity,
			2,
			true
		),
		LayerSetup(
			(Integer index) => "y``index``",
			(Integer index) => if (index == 0) then identity else BipolarSigmoid(),
			4,
			true
		),
		LayerSetup(
			(Integer index) => "z``index``",
			(Integer index) => BipolarSigmoid()
		)
	]);
	assert (is Resource resource = `module`.resourceByPath("modified_bipolar_xor.csv"));
	value trainingInputReader = () =>
		BufferedReader(InputStreamReader(CeylonResourceInputStream(resource)));
	value trainingStrategy = PatternMemorizationTrainingStrategy(
		net, 
		trainingInputReader, 
		error);
	variable value epochs = 0;
	BackpropagationTraining(
		net,
		trainingStrategy, 
		0.4,
		(Integer link) =>
			if (link == 0) then
				NguyenWidrowWeightGenerator(net,link)
			else
				RandomWeightGenerator(-0.5, 0.5),
		object extends SupervisedTrainingAdapter<BackpropagationRule>() {
			shared actual void beforeEpoch(BackpropagationRule rule) =>
				print("epoch: ``epochs++``");
			shared actual void beforeWeightsUpdate(BackpropagationRule rule, {Float*} trainings, {Float*} targets, [Float+]|[] weights) =>
				print("before: ``weights``");
			shared actual void afterWeightsUpdate(BackpropagationRule rule, {Float*} trainings, {Float*} targets, [Float+]|[] weights) =>
				print("after: ``weights``");
		})
	.train();
	assertNetLearnOperationCombinationWithinError(
		net,
		trainingInputReader,
		error,
		trainingStrategy.loss);
}

test
void fundamentalsOfNeuralNetworksBackpropNetShouldLearnXor() {
	value error = 0.05;
	value net = LayeredNeuralNetwork([
		LayerSetup(
			(Integer index) => "x``index``",
			(Integer index) => identity,
			2,
			true
		),
		LayerSetup(
			(Integer index) => "y``index``",
			(Integer index) => if (index == 0) then identity else BipolarSigmoid(),
			4,
			true
		),
		LayerSetup(
			(Integer index) => "z``index``",
			(Integer index) => BipolarSigmoid()
		)
	]);
	assert (is Resource resource = `module`.resourceByPath("modified_bipolar_xor.csv"));
	value trainingInputReader = () =>
		BufferedReader(InputStreamReader(CeylonResourceInputStream(resource)));
	variable value epochs = 0;
	value trainingStrategy = object satisfies TrainingStrategy {
		value tir = () =>
			BufferedReader(InputStreamReader(CeylonResourceInputStream(resource)));
		
		value ts = PatternMemorizationTrainingStrategy(
			net, 
			tir, 
			error);
		
		shared actual Boolean isTrainingDone() =>
			ts.isTrainingDone() || epochs > 174;
		
		shared actual Function loss(Float target) =>
			ts.loss(target);
		
		shared actual BufferedReader trainingInputReader() =>
			tir();
		
	};
	BackpropagationTraining(
		net,
		trainingStrategy, 
		0.02,
		(Integer link) =>
			object satisfies WeightGenerator {
				shared actual Float generate(Synapse synapse) {
					if (link == 0) {
						assert (exists bias = net.bias(0));
						assert (exists x1 = net.neuron("x1"));
						assert (exists x2 = net.neuron("x2"));
						assert (exists s -> w =
							zipEntries(
								net.biasSynapsesFrom(bias)
								.chain(net.synapsesFrom(x1))
								.chain(net.synapsesFrom(x2)),
								{
									-0.3378,0.2771,0.2859,-0.3329,
									0.1970,0.3191,-0.1448,0.3594,
									0.3099,0.1904,-0.0347,-0.4861
								})
								.find((Synapse synps -> Float weight) =>
									synps == synapse));
						return w;
					}
					else {
						assert (exists bias = net.bias(1));
						assert (exists y1 = net.neuron("y1"));
						assert (exists y2 = net.neuron("y2"));
						assert (exists y3 = net.neuron("y3"));
						assert (exists y4 = net.neuron("y4"));
						assert (exists s -> w =
							zipEntries(
								net.biasSynapsesFrom(bias)
								.chain(net.synapsesFrom(y1))
								.chain(net.synapsesFrom(y2))
								.chain(net.synapsesFrom(y3))
								.chain(net.synapsesFrom(y4)),
								{
									-0.1401,
									0.4919,
									-0.2913,
									-0.3979,
									0.3581
								})
								.find((Synapse synps -> Float weight) =>
									synps == synapse));
						return w;
					}
				}
			},
		object extends SupervisedTrainingAdapter<BackpropagationRule>() {
			shared actual void beforeEpoch(BackpropagationRule rule) =>
				print("epoch: ``epochs++``");
			shared actual void beforeWeightsUpdate(BackpropagationRule rule, {Float*} trainings, {Float*} targets, [Float+]|[] weights) =>
				print("before: ``weights``");
			shared actual void afterWeightsUpdate(BackpropagationRule rule, {Float*} trainings, {Float*} targets, [Float+]|[] weights) =>
				print("after: ``weights``");
		})
	.train();
	assertNetLearnOperationCombinationWithinError(
		net,
		trainingInputReader,
		error,
		trainingStrategy.loss);
}
