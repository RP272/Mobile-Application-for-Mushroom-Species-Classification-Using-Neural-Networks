package com.example.mushroomclassifier.ui.about

import android.graphics.Color
import android.opengl.Visibility
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.cardview.widget.CardView
import androidx.core.view.isVisible
import androidx.recyclerview.widget.RecyclerView
import com.example.mushroomclassifier.R
import com.example.mushroomclassifier.data.model.LicenseData
import kotlin.reflect.KClass

class LicenseAdapter(private val items: List<LicenseData>) :
    RecyclerView.Adapter<LicenseAdapter.LicenseViewHolder>() {

    private val expandedStates = BooleanArray(items.size) { false }

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
        val licenseCardMain: CardView = itemView.findViewById(R.id.licenseCardMain)
        val resourceInformation: LinearLayout = itemView.findViewById(R.id.resourceInformation)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): LicenseViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.license_card, parent, false)
        return LicenseViewHolder(view)
    }

    override fun onBindViewHolder(holder: LicenseViewHolder, position: Int) {
        val item = items[position]

        if(item.resourceName.equals("Edibility icon")){
            print("jajo")
        }

        if(!item.resourceName.isEmpty()){
            holder.resourceName.text = item.resourceName
            holder.resourceName.isVisible = true
        }else{
            holder.resourceName.isVisible = false
        }

        if(!item.resourceLicense?.isEmpty()!!){
            holder.resourceLicense.text = "Resource license: ${item.resourceLicense}"
            holder.resourceLicense.isVisible = true
        }else{
            holder.resourceLicense.isVisible = false
        }

        if(!item.resourceURL?.isEmpty()!!){
            holder.resourceURL.text = "Resource URL: ${item.resourceURL}"
            holder.resourceURL.isVisible = true
        }else{
            holder.resourceURL.isVisible = false
        }

        if(!item.resourceSiteURL.isNullOrEmpty()){
            holder.resourceSiteURL.text = "Resource site URL: ${item.resourceSiteURL}"
            holder.resourceSiteURL.isVisible = true
        }else{
            holder.resourceSiteURL.isVisible = false
        }

        if(!item.source1URL?.isEmpty()!!){
            holder.source1URL.text = "Source 1: ${item.source1URL}"
            holder.source1URL.isVisible = true
        }else{
            holder.source1URL.isVisible = false
        }

        if(!item.source2URL?.isEmpty()!!){
            holder.source2URL.text = "Source 2: ${item.source2URL}"
            holder.source2URL.isVisible = true
        }else{
            holder.source2URL.isVisible = false
        }

        if(!item.source3URL?.isEmpty()!!){
            holder.source3URL.text = "Source 3: ${item.source3URL}"
            holder.source3URL.isVisible = true
        }else{
            holder.source3URL.isVisible = false
        }

        if(!item.source4URL?.isEmpty()!!){
            holder.source4URL.text = "Source 4: ${item.source4URL}"
            holder.source4URL.isVisible = true
        }else{
            holder.source4URL.isVisible = false
        }

        if(!item.author?.isEmpty()!!){
            holder.author.text = "Author: ${item.author}"
            holder.author.isVisible = true
        }else{
            holder.author.isVisible = false
        }

        holder.resourceInformation.isVisible = expandedStates[position]
        holder.licenseCardMain.setOnClickListener {
            expandedStates[position] = !expandedStates[position]
            notifyItemChanged(position)
        }
    }

    override fun getItemCount() = items.size
}
