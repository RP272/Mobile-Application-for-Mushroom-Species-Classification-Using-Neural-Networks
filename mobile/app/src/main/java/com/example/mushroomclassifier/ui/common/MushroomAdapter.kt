package com.example.mushroomclassifier.ui.common

import android.graphics.Color
import android.text.method.ScrollingMovementMethod
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.example.mushroomclassifier.R
import com.example.mushroomclassifier.data.model.MushroomSpecies

class MushroomAdapter(private val items: List<MushroomSpecies>) :
    RecyclerView.Adapter<MushroomAdapter.MushroomViewHolder>() {

    private val expandedStates = BooleanArray(items.size) { false }

    class MushroomViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        val name: TextView = itemView.findViewById(R.id.mushroomName)
        val edibility: TextView = itemView.findViewById(R.id.mushroomEdibility)
        val description: TextView = itemView.findViewById(R.id.mushroomDescription)
        val image: ImageView = itemView.findViewById(R.id.mushroomImage)
        val probability: TextView = itemView.findViewById(R.id.mushroomProbability)
        val infoIcon: ImageView = itemView.findViewById(R.id.imageView2)
        val additionalInfo: LinearLayout = itemView.findViewById(R.id.additionalInfo)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): MushroomViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.mushroom_card, parent, false)
        return MushroomViewHolder(view)
    }

    override fun onBindViewHolder(holder: MushroomViewHolder, position: Int) {
        // TODO: Change the color of edibility icon based on edibility type of mushroom species

        val item = items[position]

        holder.name.text = item.latinName
        holder.edibility.text = "Edibility: ${item.edibility.toString().lowercase()}"
        holder.description.text = "Description: ${item.description}"

        val resId = holder.itemView.context.resources.getIdentifier(
            item.image,
            "drawable",
            holder.itemView.context.packageName
        )
        if(resId != 0){
            holder.image.setImageResource(resId)
        }else{
            holder.image.setImageResource(R.drawable.cnv1_19)
        }

        if (item.probability != null) {
            holder.probability.text = String.format("Confidence: %.2f %%", item.probability * 100)
            holder.probability.visibility = View.VISIBLE
        } else {
            holder.probability.visibility = View.GONE
        }

        if (expandedStates[position]) {
            holder.additionalInfo.alpha = 1.0f
            holder.infoIcon.setColorFilter(Color.WHITE)
            holder.description.setOnTouchListener { v, event ->
                v.parent.requestDisallowInterceptTouchEvent(true)
                false
            }
        } else {
            holder.additionalInfo.alpha = 0.0f
            holder.infoIcon.setColorFilter(0xFF6F006)
        }

        holder.infoIcon.setOnClickListener {
            expandedStates[position] = !expandedStates[position]
            notifyItemChanged(position)
        }

        holder.description.movementMethod = ScrollingMovementMethod()
    }

    override fun getItemCount() = items.size
}