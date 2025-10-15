package com.example.mushroomclassifier.ui.classify

import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.opengl.Visibility
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.view.animation.AlphaAnimation
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.view.isVisible
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.PagerSnapHelper
import androidx.recyclerview.widget.RecyclerView
import com.example.mushroomclassifier.MainActivity
import com.example.mushroomclassifier.data.model.EdibilityEnum
import com.example.mushroomclassifier.data.model.MushroomSpecies
import com.example.mushroomclassifier.data.repository.MushroomRepository
import com.example.mushroomclassifier.databinding.FragmentClassifyBinding
import com.example.mushroomclassifier.ui.common.MushroomAdapter
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils

class ClassifyFragment : Fragment() {

    private var _binding: FragmentClassifyBinding? = null

    private val binding get() = _binding!!

    private lateinit var module: Module
    private lateinit var mushroomRepository: MushroomRepository
    private lateinit var galleryLauncher: ActivityResultLauncher<String>
    private lateinit var cameraLauncher: ActivityResultLauncher<Void?>
    private lateinit var showAnim: AlphaAnimation
    private lateinit var hideAnim: AlphaAnimation

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        showAnim = AlphaAnimation(0.0f, 1.0f)
        showAnim.duration = 500
        showAnim.fillAfter = true
        hideAnim = AlphaAnimation(1.0f, 0.0f)
        hideAnim.duration = 100
        hideAnim.fillAfter = true
        _binding = FragmentClassifyBinding.inflate(inflater, container, false)
        val root: View = binding.root

        val imageView = binding.imageView;
        imageView.drawable?.alpha = (0.05 * 255).toInt();

        val context = requireContext()

        module = (requireActivity() as MainActivity).module
        mushroomRepository = (requireActivity() as MainActivity).mushroomRepository

        setupActivityResultLaunchers()

        imageView.setOnClickListener {
            showPickDialog()
        }

        binding.predictionsRecyclerView.layoutManager =
            LinearLayoutManager(context, LinearLayoutManager.HORIZONTAL, false)

        val snapHelper = PagerSnapHelper()
        snapHelper.attachToRecyclerView(binding.predictionsRecyclerView)

        binding.rightScrollIndicator.alpha = 0.0f
        binding.leftScrollIndicator.alpha = 0.0f

        binding.predictionsRecyclerView.addOnScrollListener(object: RecyclerView.OnScrollListener() {
            override fun onScrollStateChanged(recyclerView: RecyclerView, newState: Int) {
                when (newState) {
                    RecyclerView.SCROLL_STATE_IDLE -> {
                        val layoutManager = recyclerView.layoutManager as LinearLayoutManager
                        val firstVisible = layoutManager.findFirstCompletelyVisibleItemPosition()
                        val lastVisible = layoutManager.findLastCompletelyVisibleItemPosition()

                        val itemCount = recyclerView.adapter?.itemCount ?: 0

                        if(firstVisible > 0){
                            binding.leftScrollIndicator.startAnimation(showAnim)
                        }

                        if (lastVisible < itemCount - 1) {
                            binding.rightScrollIndicator.startAnimation(showAnim)
                        }
                    }
                    RecyclerView.SCROLL_STATE_DRAGGING -> {
                        binding.leftScrollIndicator.startAnimation(hideAnim)
                        binding.rightScrollIndicator.startAnimation(hideAnim)
                    }
                }
            }
        })

        binding.textClassify.setText("UPLOAD PHOTO")
        return root
    }

    private fun showPickDialog() {
        val options = arrayOf("Gallery", "Camera")

        android.app.AlertDialog.Builder(requireContext())
            .setTitle("Pick image")
            .setItems(options) { _, which ->
                when (which) {
                    0 -> galleryLauncher.launch("image/*")
                    1 -> cameraLauncher.launch(null)
                }
            }
            .show()
    }

    private fun setupActivityResultLaunchers() {
        // Pick from gallery
        galleryLauncher = registerForActivityResult(
            ActivityResultContracts.GetContent()
        ) { uri: Uri? ->
            uri?.let { handleImageUri(it) }
        }

        // Capture from camera
        cameraLauncher = registerForActivityResult(
            ActivityResultContracts.TakePicturePreview()
        ) { bitmap: Bitmap? ->
            bitmap?.let { runModel(it) }
        }
    }

    private fun handleImageUri(uri: Uri) {
        val bitmap: Bitmap = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            // API 28+ (Android 9+)
            val source = ImageDecoder.createSource(requireContext().contentResolver, uri)
            ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
                decoder.isMutableRequired = true
            }
        } else {
            // Fallback for older devices
            @Suppress("DEPRECATION")
            MediaStore.Images.Media.getBitmap(requireContext().contentResolver, uri)
        }
        runModel(bitmap)
    }

    private fun runModel(bitmap: Bitmap) {
        binding.imageView.setImageBitmap(bitmap)

        val tensor = bitmapToTensor(bitmap)
        val outputTensor = module.forward(IValue.from(tensor)).toTensor()
        val scores = outputTensor.dataAsFloatArray

        // Softmax
        val expScores = scores.map { kotlin.math.exp(it) }
        val sumExp = expScores.sum()
        val probabilities = expScores.map { it / sumExp }

        val maxIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: -1
        val maxProb = if (maxIndex != -1) probabilities[maxIndex] else 0f

        val top3 = probabilities
            .withIndex()
            .sortedByDescending { it.value }
            .take(3)

        val predictions = top3.map { (index, prob) ->
            // TODO: Move loading the species data to MainActivity
            val species = mushroomRepository.getSpeciesByIndex(requireContext(), index)

            MushroomSpecies(
                latinName = species?.latinName ?: "Unknown",
                image = species?.image ?: "@drawable/pexels_ekamelev_4178330",
                description = species?.description ?: "Description",
                edibility = species?.edibility ?: EdibilityEnum.INEDIBLE,
                probability = prob
            )
        }

        binding.textClassify.visibility = View.GONE
        binding.imageView.visibility = View.GONE
        binding.predictionsRecyclerView.visibility = View.VISIBLE
        binding.predictionsRecyclerView.scrollToPosition(0)
        binding.predictionsRecyclerView.adapter = MushroomAdapter(predictions)
        binding.rightScrollIndicator.alpha = 1.0f
        binding.leftScrollIndicator.alpha = 1.0f
        binding.leftScrollIndicator.startAnimation(hideAnim)
    }

    private fun bitmapToTensor(bitmap: Bitmap): Tensor {
        val resized = Bitmap.createScaledBitmap(bitmap, 256, 256, true)
        return TensorImageUtils.bitmapToFloat32Tensor(
            resized,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
            TensorImageUtils.TORCHVISION_NORM_STD_RGB
        )
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
