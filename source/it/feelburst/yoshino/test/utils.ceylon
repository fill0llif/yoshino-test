import ceylon.random {
	Random
}
Float nextExclusiveFloat(Random random) =>
	let (rn = random.nextFloat())
	if (rn == 0.0) then nextExclusiveFloat(random)
	else rn;