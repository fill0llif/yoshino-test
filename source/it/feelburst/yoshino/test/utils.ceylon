import ceylon.random {
	Random
}
shared Float nextExclusiveFloat(Random random) =>
	let (rn = random.nextFloat())
	if (rn == 0.0) then nextExclusiveFloat(random)
	else rn;