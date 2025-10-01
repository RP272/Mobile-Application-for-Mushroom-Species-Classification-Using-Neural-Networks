package com.example.mushroomclassifier.data.model

import kotlinx.serialization.Serializable

@Serializable
data class MushroomSpecies(
    val latinName: String,
    val description: String,
    val edibility: EdibilityEnum,
    val image: String,
    val probability: Float? = null
)