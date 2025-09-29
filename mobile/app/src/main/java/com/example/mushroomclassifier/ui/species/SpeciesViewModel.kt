package com.example.mushroomclassifier.ui.species

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel

class SpeciesViewModel : ViewModel() {

    private val _text = MutableLiveData<String>().apply {
        value = "This is species Fragment"
    }
    val text: LiveData<String> = _text
}