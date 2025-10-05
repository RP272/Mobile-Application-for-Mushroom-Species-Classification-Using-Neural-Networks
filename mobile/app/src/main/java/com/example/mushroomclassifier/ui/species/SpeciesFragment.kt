package com.example.mushroomclassifier.ui.species

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.SearchView
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.PagerSnapHelper
import com.example.mushroomclassifier.MainActivity
import com.example.mushroomclassifier.data.model.MushroomSpecies
import com.example.mushroomclassifier.data.repository.MushroomRepository
import com.example.mushroomclassifier.databinding.FragmentSpeciesBinding
import com.example.mushroomclassifier.ui.common.MushroomAdapter

class SpeciesFragment : Fragment() {

    private var _binding: FragmentSpeciesBinding? = null
    private lateinit var mushroomRepository: MushroomRepository
    private var speciesTotalCount: Int = 0

    // This property is only valid between onCreateView and
    // onDestroyView.
    private val binding get() = _binding!!

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        val notificationsViewModel =
            ViewModelProvider(this).get(SpeciesViewModel::class.java)

        _binding = FragmentSpeciesBinding.inflate(inflater, container, false)

        binding.speciesRecyclerView.layoutManager =
            LinearLayoutManager(requireContext(), LinearLayoutManager.VERTICAL, false)

        mushroomRepository = (requireActivity() as MainActivity).mushroomRepository
        val species = mushroomRepository.getAllSpecies(requireContext()).map { s ->
            MushroomSpecies(
                latinName = s.latinName,
                image = s.image,
                description = s.description,
                edibility = s.edibility,
                probability = null
            )
        }
        speciesTotalCount = species.count()
        binding.speciesCounter.text = "$speciesTotalCount / $speciesTotalCount"
        binding.speciesRecyclerView.adapter = MushroomAdapter(species)
        val snapHelper = PagerSnapHelper()
        snapHelper.attachToRecyclerView(binding.speciesRecyclerView)

        binding.speciesSearchView.setOnQueryTextListener(object : SearchView.OnQueryTextListener {
            override fun onQueryTextChange(newText: String?): Boolean {
                if (newText != null) {
                    val filteredSpecies = species.filter { e ->
                        e.latinName.lowercase().contains(newText.lowercase())
                    }
                    binding.speciesRecyclerView.adapter = MushroomAdapter(filteredSpecies)
                    val filteredCount = filteredSpecies.count()
                    binding.speciesCounter.text = "$filteredCount / $speciesTotalCount"
                }
                return true
            }

            override fun onQueryTextSubmit(query: String?): Boolean {
                if (query != null) {
                    val filteredSpecies = species.filter { e ->
                        e.latinName.lowercase().contains(query.lowercase())
                    }
                    binding.speciesRecyclerView.adapter = MushroomAdapter(filteredSpecies)
                    val filteredCount = filteredSpecies.count()
                    binding.speciesCounter.text = "$filteredCount / $speciesTotalCount"
                }
                return true
            }
        })

        val root: View = binding.root
        return root
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}