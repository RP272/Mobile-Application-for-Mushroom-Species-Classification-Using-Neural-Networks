package com.example.mushroomclassifier.ui.common

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.example.mushroomclassifier.data.model.MushroomSpecies
import com.example.mushroomclassifier.R

class MushroomAdapter(private val items: List<MushroomSpecies>) :
    RecyclerView.Adapter<MushroomAdapter.MushroomViewHolder>() {
    class MushroomViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        val name: TextView = itemView.findViewById(R.id.mushroomName)
        val edibility: TextView = itemView.findViewById(R.id.mushroomEdibility)
        val description: TextView = itemView.findViewById(R.id.mushroomDescription)
        val image: ImageView = itemView.findViewById(R.id.mushroomImage)
        val probability: TextView = itemView.findViewById(R.id.mushroomProbability)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): MushroomViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.mushroom_card, parent, false)
        return MushroomViewHolder(view)
    }

    override fun onBindViewHolder(holder: MushroomViewHolder, position: Int) {
        val item = items[position]

        holder.name.text = item.latinName
        holder.edibility.text = item.edibility
        holder.description.text = item.description

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
            holder.probability.text = String.format("Confidence: %.2f%%", item.probability * 100)
            holder.probability.visibility = View.VISIBLE
        } else {
            holder.probability.visibility = View.GONE
        }
    }

    override fun getItemCount() = items.size
}