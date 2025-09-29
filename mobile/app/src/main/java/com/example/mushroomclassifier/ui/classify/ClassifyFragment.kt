package com.example.mushroomclassifier.ui.classify

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import com.example.mushroomclassifier.databinding.FragmentClassifyBinding

class ClassifyFragment : Fragment() {

    private var _binding: FragmentClassifyBinding? = null

    // This property is only valid between onCreateView and
    // onDestroyView.
    private val binding get() = _binding!!

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        val classifyViewModel =
            ViewModelProvider(this).get(ClassifyViewModel::class.java)

        _binding = FragmentClassifyBinding.inflate(inflater, container, false)
        val root: View = binding.root

        val textView: TextView = binding.textClassify
        classifyViewModel.text.observe(viewLifecycleOwner) {
            textView.text = it
        }

        val imageView = binding.imageView;
        imageView.drawable?.alpha = (0.05 * 255).toInt();

        imageView.setOnClickListener {
            // TODO: photo upload + model inference
            Toast.makeText(requireContext(), "CAMERA BUTTON CLICKED", Toast.LENGTH_SHORT).show();
        }
        return root
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}