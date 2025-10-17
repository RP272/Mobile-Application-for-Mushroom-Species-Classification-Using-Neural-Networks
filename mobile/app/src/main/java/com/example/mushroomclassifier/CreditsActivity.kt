package com.example.mushroomclassifier

import android.os.Bundle
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.mushroomclassifier.data.model.LicenseData
import com.example.mushroomclassifier.data.repository.LicenseRepository
import com.example.mushroomclassifier.data.repository.MushroomRepository
import com.example.mushroomclassifier.databinding.ActivityCreditsBinding
import com.example.mushroomclassifier.databinding.ActivityMainBinding
import com.example.mushroomclassifier.ui.about.LicenseAdapter

class CreditsActivity : AppCompatActivity() {
    private lateinit var binding: ActivityCreditsBinding
    private lateinit var licenseRepository: LicenseRepository

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityCreditsBinding.inflate(layoutInflater)

        setContentView(binding.root)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.activityMainLayout)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        supportActionBar?.hide()

        licenseRepository = LicenseRepository()
        val licenses = licenseRepository.getAllLicenses(this)
        binding.licenseRecyclerView.layoutManager =
        LinearLayoutManager(this, LinearLayoutManager.VERTICAL, false)
        binding.licenseRecyclerView.adapter = LicenseAdapter(licenses)
    }
}