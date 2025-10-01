package com.example.mushroomclassifier.data.repository

import android.content.Context
import android.util.Log
import com.example.mushroomclassifier.R
import com.example.mushroomclassifier.data.model.MushroomSpecies
import kotlinx.serialization.json.Json

class MushroomRepository {
    private var speciesList: List<MushroomSpecies>? = null

    fun loadData(context: Context): List<MushroomSpecies> {
        if(speciesList === null){
            val inputStream = context.resources.openRawResource(R.raw.mushrooms)
            val json = inputStream.bufferedReader().use { it.readText() }
            val format = Json { decodeEnumsCaseInsensitive = true }
            speciesList = format.decodeFromString<List<MushroomSpecies>>(json)
        }
        return speciesList!!
    }

    fun getSpeciesByIndex(context: Context, index: Int): MushroomSpecies? {
        return loadData(context).getOrNull(index)
    }

    fun getAllSpecies(context: Context): List<MushroomSpecies> {
        return loadData(context)
    }
}