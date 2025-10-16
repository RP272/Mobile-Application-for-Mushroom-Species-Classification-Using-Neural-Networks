package com.example.mushroomclassifier.ui.about

import android.content.Intent
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import com.example.mushroomclassifier.CreditsActivity
import com.example.mushroomclassifier.databinding.FragmentAboutBinding

class AboutFragment : Fragment() {

    private var _binding: FragmentAboutBinding? = null

    // This property is only valid between onCreateView and
    // onDestroyView.
    private val binding get() = _binding!!

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentAboutBinding.inflate(inflater, container, false)
        val root: View = binding.root

        binding.textAbout.setText("FUNGOOSE is a mobile application for mushroom species classification using convolutional neural networks (CNN). The application can be used to identify 106 mushroom species that are included in the list of MO106 dataset (Mushroom Observer 106). The convolutional neural network of choice is MobileNetV4.")

        binding.textAuthor.setText("Created by: Robert Pytel")

        binding.creditsButtons.setOnClickListener {
            startActivity(Intent(requireContext(), CreditsActivity::class.java))
        }
        return root
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}