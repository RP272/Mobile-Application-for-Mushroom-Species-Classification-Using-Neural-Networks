package com.example.mushroomclassifier.data.model;

import kotlinx.serialization.SerialName;
import kotlinx.serialization.Serializable;

@Serializable
public enum EdibilityEnum {
    EDIBLE,
    INEDIBLE,
    TOXIC
}
