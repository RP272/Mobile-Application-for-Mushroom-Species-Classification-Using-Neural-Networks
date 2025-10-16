package com.example.mushroomclassifier.data.repository

import android.content.Context
import com.example.mushroomclassifier.R
import com.example.mushroomclassifier.data.model.LicenseData
import com.example.mushroomclassifier.data.model.MushroomSpecies
import kotlinx.serialization.json.Json

class LicenseRepository {
    private var licenseList: List<LicenseData>? = null

    fun loadData(context: Context): List<LicenseData> {
        if(licenseList === null){
            val inputStream = context.resources.openRawResource(R.raw.licenses)
            val json = inputStream.bufferedReader().use { it.readText() }
            val format = Json { decodeEnumsCaseInsensitive = true }
            licenseList = format.decodeFromString<List<LicenseData>>(json)
        }
        return licenseList!!
    }

    fun getAllLicenses(context: Context): List<LicenseData> {
        return loadData(context)
    }
}