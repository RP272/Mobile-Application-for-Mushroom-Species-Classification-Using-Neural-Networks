package com.example.mushroomclassifier.ui.classify

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.PagerSnapHelper
import com.example.mushroomclassifier.data.model.MushroomSpecies
import com.example.mushroomclassifier.databinding.FragmentClassifyBinding
import com.example.mushroomclassifier.ui.common.MushroomAdapter
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream

class ClassifyFragment : Fragment() {

    private var _binding: FragmentClassifyBinding? = null

    // This property is only valid between onCreateView and
    // onDestroyView.
    private val binding get() = _binding!!

    private lateinit var module: Module
    private lateinit var galleryLauncher: ActivityResultLauncher<String>
    private lateinit var cameraLauncher: ActivityResultLauncher<Void?>

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

        val context = requireContext()

        // Loading the model in this place slows down the classify page open. TODO: Move module loading to MainActivity
        module = LiteModuleLoader.load(
            assetFilePath(
                context,
                "MobileNetV4-Mushroom-Classifier-MO106.ptl"
            )
        )

        setupActivityResultLaunchers()

        imageView.setOnClickListener {
            showPickDialog()
        }

        binding.predictionsRecyclerView.layoutManager =
            LinearLayoutManager(requireContext(), LinearLayoutManager.HORIZONTAL, false)

        val snapHelper = PagerSnapHelper()
        snapHelper.attachToRecyclerView(binding.predictionsRecyclerView)

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
            MushroomSpecies(
                latinName = mushroomLabels.getOrNull(index) ?: "Unknown",
                image = "@drawable/pexels_ekamelev_4178330",
                description = "some description",
                edibility = "Unknown",
                probability = prob
            )
        }

        binding.textClassify.visibility = View.GONE
        binding.imageView.visibility = View.GONE
        binding.predictionsRecyclerView.visibility = View.VISIBLE
        binding.predictionsRecyclerView.adapter = MushroomAdapter(predictions)
        binding.predictionsRecyclerView.scrollToPosition(0)
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

    fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (!file.exists()) {
            context.assets.open(assetName).use { inputStream ->
                FileOutputStream(file).use { outputStream ->
                    val buffer = ByteArray(4 * 1024)
                    var read: Int
                    while (inputStream.read(buffer).also { read = it } != -1) {
                        outputStream.write(buffer, 0, read)
                    }
                    outputStream.flush()
                }
            }
        }
        return file.absolutePath
    }

    private val mushroomLabels = listOf(
        "Agaricus augustus",
        "Agaricus xanthodermus",
        "Amanita amerirubescens",
        "Amanita augusta",
        "Amanita brunnescens",
        "Amanita calyptroderma",
        "Amanita flavoconia",
        "Amanita muscaria",
        "Amanita persicina",
        "Amanita phalloides",
        "Amanita velosa",
        "Armillaria mellea",
        "Armillaria tabescens",
        "Artomyces pyxidatus",
        "Bolbitius titubans",
        "Boletus pallidus",
        "Boletus rex-veris",
        "Cantharellus californicus",
        "Cantharellus cinnabarinus",
        "Cerioporus squamosus",
        "Chlorophyllum brunneum",
        "Chlorophyllum molybdites",
        "Clitocybe nuda",
        "Coprinellus micaceus",
        "Coprinopsis lagopus",
        "Coprinus comatus",
        "Crucibulum laeve",
        "Cryptoporus volvatus",
        "Daedaleopsis confragosa",
        "Entoloma abortivum",
        "Flammulina velutipes",
        "Fomitopsis mounceae",
        "Galerina marginata",
        "Ganoderma applanatum",
        "Ganoderma curtisii",
        "Ganoderma oregonense",
        "Ganoderma tsugae",
        "Gliophorus psittacinus",
        "Gloeophyllum sepiarium",
        "Grifola frondosa",
        "Gymnopilus luteofolius",
        "Hericium coralloides",
        "Hericium erinaceus",
        "Hygrophoropsis aurantiaca",
        "Hypholoma fasciculare",
        "Hypholoma lateritium",
        "Hypomyces lactifluorum",
        "Ischnoderma resinosum",
        "Laccaria ochropurpurea",
        "Lacrymaria lacrymabunda",
        "Lactarius indigo",
        "Laetiporus sulphureus",
        "Laricifomes officinalis",
        "Leratiomyces ceres",
        "Leucoagaricus americanus",
        "Leucoagaricus leucothites",
        "Lycogala epidendrum",
        "Lycoperdon perlatum",
        "Lycoperdon pyriforme",
        "Mycena haematopus",
        "Mycena leaiana",
        "Omphalotus illudens",
        "Omphalotus olivascens",
        "Panaeolina foenisecii",
        "Panaeolus cinctulus",
        "Panaeolus papilionaceus",
        "Panellus stipticus",
        "Phaeolus schweinitzii",
        "Phlebia tremellosa",
        "Phyllotopsis nidulans",
        "Pleurotus ostreatus",
        "Pleurotus pulmonarius",
        "Pluteus cervinus",
        "Psathyrella candolleana",
        "Pseudohydnum gelatinosum",
        "Psilocybe allenii",
        "Psilocybe aztecorum",
        "Psilocybe azurescens",
        "Psilocybe caerulescens",
        "Psilocybe cubensis",
        "Psilocybe cyanescens",
        "Psilocybe muliercula",
        "Psilocybe neoxalapensis",
        "Psilocybe ovoideocystidiata",
        "Psilocybe pelliculosa",
        "Psilocybe zapotecorum",
        "Retiboletus ornatipes",
        "Sarcomyxa serotina",
        "Schizophyllum commune",
        "Stereum ostrea",
        "Stropharia ambigua",
        "Stropharia rugosoannulata",
        "Suillus americanus",
        "Suillus luteus",
        "Suillus spraguei",
        "Tapinella atrotomentosa",
        "Trametes betulina",
        "Trametes gibbosa",
        "Trametes versicolor",
        "Trichaptum biforme",
        "Tricholoma murrillianum",
        "Tricholomopsis rutilans",
        "Tubaria furfuracea",
        "Tylopilus felleus",
        "Tylopilus rubrobrunneus",
        "Volvopluteus gloiocephalus"
    )
}
