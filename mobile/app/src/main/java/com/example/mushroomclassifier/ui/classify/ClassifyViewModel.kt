package com.example.mushroomclassifier.ui.classify

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel

class ClassifyViewModel : ViewModel() {

    private val _text = MutableLiveData<String>().apply {
        value = "This is classify Fragment"
    }
    val text: LiveData<String> = _text
}