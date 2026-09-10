package com.swarved.guard.model

data class ScamIncident(
    val timestamp: String,
    val callerId: String,
    val syntheticProbability: Int
)
