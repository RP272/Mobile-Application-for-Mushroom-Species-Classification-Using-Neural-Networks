package com.example.mushroomclassifier

import android.os.Bundle
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.example.mushroomclassifier.data.repository.LicenseRepository
import com.example.mushroomclassifier.data.repository.MushroomRepository

class CreditsActivity : AppCompatActivity() {
    private lateinit var licenseRepository: LicenseRepository

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_credits)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        supportActionBar?.hide()

        licenseRepository = LicenseRepository()
        val licenses = licenseRepository.getAllLicenses(this)
    
    }
}