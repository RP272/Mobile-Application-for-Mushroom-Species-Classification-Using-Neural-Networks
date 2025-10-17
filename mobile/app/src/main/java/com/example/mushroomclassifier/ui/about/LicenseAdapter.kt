package com.example.mushroomclassifier.ui.about

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.example.mushroomclassifier.R
import com.example.mushroomclassifier.data.model.LicenseData

class LicenseAdapter(private val items: List<LicenseData>) :
    RecyclerView.Adapter<LicenseAdapter.LicenseViewHolder>() {
    class LicenseViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        val resourceName: TextView = itemView.findViewById(R.id.resourceName)
        val resourceURL: TextView = itemView.findViewById(R.id.resourceURL)
        val resourceLicense: TextView = itemView.findViewById(R.id.resourceLicense)
        val licenseURL: TextView = itemView.findViewById(R.id.licenseURL)
        val resourceSiteURL: TextView = itemView.findViewById(R.id.resourceSiteURL)
        val source1URL: TextView = itemView.findViewById(R.id.source1URL)
        val source2URL: TextView = itemView.findViewById(R.id.source2URL)
        val source3URL: TextView = itemView.findViewById(R.id.source3URL)
        val source4URL: TextView = itemView.findViewById(R.id.source4URL)
        val author: TextView = itemView.findViewById(R.id.author)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): LicenseViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.license_card, parent, false)
        return LicenseViewHolder(view)
    }

    override fun onBindViewHolder(holder: LicenseViewHolder, position: Int) {
        val item = items[position]

        holder.resourceName.text = item.resourceName
        holder.resourceURL.text = "Resource URL: ${item.resourceURL}"
        holder.resourceLicense.text = "Resource license: ${item.resourceLicense}"
        holder.resourceSiteURL.text = "Resource site URL: ${item.resourceSiteURL}"
        holder.source1URL.text = "Source 1: ${item.source1URL}"
        holder.source2URL.text = "Source 2: ${item.source2URL}"
        holder.source3URL.text = "Source 3: ${item.source3URL}"
        holder.source4URL.text = "Source 4: ${item.source4URL}"
        holder.author.text = "Author: ${item.author}"
    }

    override fun getItemCount() = items.size
}
