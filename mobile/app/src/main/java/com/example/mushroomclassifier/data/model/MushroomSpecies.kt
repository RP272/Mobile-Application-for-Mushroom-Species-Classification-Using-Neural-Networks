package com.example.mushroomclassifier.data.model

data class MushroomSpecies(
    val latinName: String,
    val description: String,
    val edibility: String, // TODO: enum
    val image: String,
    val probability: Float? = null
)