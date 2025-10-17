package com.example.mushroomclassifier.data.model

import kotlinx.serialization.Serializable

@Serializable
data class LicenseData(
    val resourceName: String,
    val resourceURL: String,
    val resourceLicense: String?,
    val licenseURL: String?,
    val resourceSiteURL: String?,
    val source1URL: String?,
    val source2URL: String?,
    val source3URL: String?,
    val source4URL: String?,
    val author: String?
)