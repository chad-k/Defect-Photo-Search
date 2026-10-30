from pathlib import Path
import csv
import io
import streamlit as st
from demo_data import ensure_demo, TYPES
from engine import read_image, png_bytes, load_cases, image_path, search, add_case

st.set_page_config(page_title='Defect Photo Search | Hertzler demo',page_icon='🔎',layout='wide')
ROOT=Path(__file__).resolve().parent
try:
    ensure_demo(ROOT/'demo_realistic')
except (OSError, ValueError) as exc:
    st.error(str(exc))
    st.stop()
demo_rows=load_cases(ROOT/'demo_realistic')
# Use Streamlit's native theme for all surfaces and text.
# Hardcoded light backgrounds make dark-theme text unreadable.
st.caption('HERTZLER SYSTEMS  /  LOCAL PROTOTYPE')
st.title('Defect Photo Search')
st.write('Find visually similar cases. Review the evidence behind each match.')
with st.sidebar:
    st.header('Search workspace')
    dataset=st.radio('Case collection',['Demonstration','My local cases'])
    root=ROOT/('demo_realistic' if dataset=='Demonstration' else 'local_cases')
    method=st.selectbox('Search method',['Demo visual features','CLIP pretrained model'],index=1 if (ROOT/'models'/'clip'/'config.json').exists() else 0)
    st.caption('Demo visual features uses handcrafted patterns, not a pretrained AI model. CLIP requires optional setup.')
    topk=st.slider('Maximum matches',1,12,6)
    st.divider()
    st.write('**Local processing**')
    st.caption('No cloud search or photo-upload service. Model downloads are a separate, explicit setup step.')
    st.caption('Visual similarity is not a defect diagnosis or proof of a shared cause.')
try: rows=load_cases(root)
except Exception as exc:
    st.error(f'Cannot read the case collection: {exc}'); st.stop()
if dataset=='Demonstration':
    st.info('DEMONSTRATION ONLY · 100 photorealistic AI-generated reference images, separate from the 100-image test pack. All case histories are fictional. Results do not establish real-world search accuracy.')
a,b,c=st.columns(3)
a.metric('Historical images',len(rows))
b.metric('Collection','Photorealistic synthetic' if dataset=='Demonstration' else 'Your local records')
c.metric('Processing','On this computer')
search_tab,browse_tab,add_tab,help_tab=st.tabs(['Search photos','Browse cases','Add a case','How it works'])
with search_tab:
    parts=sorted({r.get('part','Unknown') for r in rows})
    selected=st.multiselect('Limit search to parts',parts,help='Leave empty to search all parts. Defect labels are never used to rank images.')
    available=[r for r in rows if not selected or r.get('part') in selected]
    source=st.radio('Query source',['Try a demo image','Upload a photo'],horizontal=True)
    im=None
    if source=='Try a demo image':
        kind=st.selectbox('Example pattern',TYPES)
        examples=[r for r in demo_rows if r['defect']==kind]
        example_id=st.selectbox('Reference example',[r['case_id'] for r in examples])
        example=next(r for r in examples if r['case_id']==example_id)
        im=read_image(image_path(ROOT/'demo_realistic',example))
        st.caption('This sample comes from the reference collection. Its exact image is always excluded from results. For independent queries, upload an image from your separate 100-image test pack.')
    else:
        uploaded=st.file_uploader('Photo to investigate',type=['png','jpg','jpeg','webp'],key='query_upload')
        if uploaded:
            try: im=read_image(uploaded.getvalue())
            except Exception as exc: st.error(f'Cannot open photo: {exc}')
    if im is not None:
        query=im
        preview,controls=st.columns([1,2])
        with controls:
            crop=st.checkbox('Crop to the area of interest')
            if crop:
                x=st.slider('Horizontal range (%)',0,100,(0,100))
                y=st.slider('Vertical range (%)',0,100,(0,100))
                if x[0]>=x[1] or y[0]>=y[1]:
                    st.error('Select a nonzero crop width and height.'); st.stop()
                query=im.crop((int(im.width*x[0]/100),int(im.height*y[0]/100),max(1,int(im.width*x[1]/100)),max(1,int(im.height*y[1]/100))))
                if query.width<8 or query.height<8:
                    st.error('Choose a larger crop (at least 8 × 8 pixels).'); st.stop()
                st.caption('Crop the same type of area in reference photos for meaningful comparisons.')
            exclude=st.checkbox('Exclude exact copies of the original query',value=True,disabled=source=='Try a demo image')
            if source=='Try a demo image': exclude=True
            st.write(f'**{len(available)} reference images before duplicate exclusion**')
            st.caption('Scores are raw cosine similarity (−1 to 1), not probabilities. Even an unrelated query will have a highest-ranked result; there is no validated acceptance threshold.')
        with preview:
            st.image(query,caption='Image being searched',use_container_width=True)
        if st.button('Find similar cases',type='primary',disabled=not available):
            try:
                with st.spinner('Comparing images locally…'):
                    result,errors=search(query,available,root,method,ROOT/'models'/'clip',png_bytes(im) if exclude else None)
                st.session_state['result']=(result[:topk],errors,png_bytes(query),dataset,method,tuple(selected),topk,png_bytes(im),exclude)
            except Exception as exc:
                st.session_state.pop('result',None)
                st.error(f'Search could not run: {exc}')
                st.info('For CLIP, install requirements-ai.txt and run download_model.py once while online. Demo visual features works without model downloads.')
        saved=st.session_state.get('result')
        signature=(dataset,method,tuple(selected),topk,png_bytes(im),exclude)
        if saved and saved[3:]==signature and saved[2]==png_bytes(query):
            result,errors=saved[:2]
            for err in errors: st.warning(f'Skipped unreadable reference: {err}')
            st.subheader('Similar historical cases')
            if not result: st.warning('No reference images remain after filtering or duplicate exclusion.')
            st.caption('Ranked by image content only. Notes are retrieved from the matching record; they are not AI-generated explanations of your photo.')
            for start in range(0,len(result),3):
                cols=st.columns(3)
                for col,(score,row) in zip(cols,result[start:start+3]):
                    with col:
                        with st.container(border=True):
                            st.image(str(image_path(root,row)),use_container_width=True)
                            st.markdown(f"**{row['case_id']} · {row.get('defect','Unclassified')}**")
                            st.write(f'Visual similarity: **{score:.3f}**')
                            st.caption(f"{row.get('part','')} · {row.get('machine','')}")
                            st.write(row.get('notes',''))
                            with st.expander('Investigation record'):
                                st.write('**Status:**',row.get('status','Unverified'))
                                st.write('**Recorded cause / hypothesis:**',row.get('cause','Not recorded'))
                                st.write('**Recorded action:**',row.get('action','Not recorded'))
                                st.caption('Historical actions need independent review before use on a new case.')
            if result:
                buf=io.StringIO(); fields=['rank','similarity','case_id','part','defect','status','notes','cause','action']
                writer=csv.DictWriter(buf,fieldnames=fields); writer.writeheader()
                for rank,(score,row) in enumerate(result,1):
                    out={key:row.get(key,'') for key in fields}; out.update(rank=rank,similarity=round(score,4))
                    out={k: ("'"+v if isinstance(v,str) and v.startswith(('=','+','-','@')) else v) for k,v in out.items()}
                    writer.writerow(out)
                st.download_button('Download these matches as CSV',buf.getvalue(),'defect_matches.csv','text/csv')
    if not available: st.info('No cases available. Select Demonstration or add a local case.')
with browse_tab:
    st.caption('Stored labels and notes describe reference cases. They are not used for visual ranking.')
    label=st.selectbox('Browse by recorded defect',['All']+sorted({r.get('defect','Unclassified') for r in rows}))
    browsed=[r for r in rows if label=='All' or r.get('defect')==label]
    page=st.number_input('Page',min_value=1,max_value=max(1,(len(browsed)+11)//12),value=1,step=1)
    for offset in range((page-1)*12,min(page*12,len(browsed)),4):
        for col,row in zip(st.columns(4),browsed[offset:offset+4]):
            try: col.image(str(image_path(root,row)),caption=f"{row['case_id']} · {row.get('defect','')}",use_container_width=True)
            except Exception as exc: col.warning(str(exc))
with add_tab:
    st.write('Add a reference photo and its known history to **My local cases**. Synthetic examples remain separate.')
    st.caption('Single-user prototype. For multiple simultaneous writers, use a database-backed import workflow.')
    with st.form('add_case',clear_on_submit=True):
        upload=st.file_uploader('Reference photo',type=['png','jpg','jpeg','webp'],key='reference_upload')
        p1,p2=st.columns(2)
        part=p1.text_input('Part number / product *')
        defect=p2.text_input('Recorded defect label *')
        machine=p1.text_input('Machine / line')
        material=p2.text_input('Material')
        notes=st.text_area('Observed condition')
        cause=st.text_area('Recorded cause or hypothesis')
        action=st.text_area('Recorded action and outcome')
        status=st.selectbox('Investigation status',['Unverified','Under investigation','Verified by your team'])
        submit=st.form_submit_button('Save local reference case')
        if submit:
            if not upload or not part.strip() or not defect.strip(): st.error('Photo, part and defect label are required.')
            else:
                try:
                    cid=add_case(ROOT/'local_cases',read_image(upload.getvalue()),dict(part=part.strip(),defect=defect.strip(),machine=machine,material=material,notes=notes,cause=cause,action=action,status=status))
                    st.success(f'Saved {cid}. Choose My local cases in the sidebar to search it.')
                except Exception as exc: st.error(f'Could not save case: {exc}')
with help_tab:
    st.markdown('''### Try the demonstration
1. Keep **Demonstration** selected. Use **CLIP pretrained model** if installed.
2. Upload a photo from the separate test pack, or choose a reference example (its own image is excluded), then click **Find similar cases**.
3. Compare images and expand investigation records.
4. Try part filters and cropping; inspect how rankings change.

The collection has 100 new references, 10 in each of 10 categories. None of the separate 100 test-pack images are used as references. All photos are AI-generated; intended defect labels and case narratives are illustrative, not real inspection evidence.

### Two search methods
**Demo visual features:** small, fast, handcrafted descriptors of shape, edges, contrast and color. No AI weights are needed. Suitable for demonstrating workflow; not validated for real defect search.

**CLIP pretrained model:** an optional general-purpose neural image encoder running on CPU. It is not trained specifically on manufacturing defects. Install the optional packages and explicitly download the model once. Search itself only loads local files and fails rather than downloading missing weights.

### Scores and limitations
Cosine similarity is a ranking signal, not confidence, classification accuracy, or root-cause probability. Scores from different methods should not be compared. Backgrounds, lighting, product geometry and crop scale can dominate matching. No automatic defect diagnosis or release decision is made.

### Your images
Use Add a case to build a separate local collection. Imported images are converted to PNG with metadata removed. New query uploads are held in application memory/cache, not added to the collection. Restart to clear in-memory caches. Saved reference images and notes remain in local_cases. Back up that folder. This prototype has no authentication and is intended for one local user.

### Before real use
Test with representative, independently labeled photos, include unrelated queries, exclude duplicates across evaluation sets, and measure retrieval relevance. Real-world performance has not been established by this synthetic demo.
''')
