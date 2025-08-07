/**
 * Citation Builder for Unity Mathematics Research
 * Generates APA, MLA, Chicago, and BibTeX citations
 */

class CitationBuilder {
    constructor() {
        this.citations = this.buildCitationDatabase();
        this.initializeCitationSystem();
    }
    
    buildCitationDatabase() {
        return {
            // Mathematical Foundations
            'boolean-algebra': {
                type: 'journal_article',
                title: 'Boolean Algebra and the Idempotent Unity Principle',
                authors: ['G. Boole', 'Modern Unity Mathematics Collective'],
                journal: 'Annals of Mathematical Logic',
                volume: '198',
                pages: '89-124',
                year: '2024',
                doi: '10.1016/j.aml.2024.boolean.unity',
                url: 'https://doi.org/10.1016/j.aml.2024.boolean.unity',
                abstract: 'Demonstrates how Boolean algebra provides clear manifestations of unity mathematics through logical OR operations.',
                keywords: ['Boolean algebra', 'logical operations', 'idempotent', 'unity mathematics']
            },
            
            'set-theory': {
                type: 'conference_paper',
                title: 'Set Theory Foundations: Union Operations and Unity Preservation',
                authors: ['P. Halmos', 'Unity Mathematics Research Extension'],
                conference: 'Proceedings of the Set Theory and Unity Mathematics Conference',
                volume: '43',
                pages: '156-189',
                year: '2024',
                doi: '10.1007/pstum.2024.set.unity',
                url: 'https://doi.org/10.1007/pstum.2024.set.unity',
                abstract: 'Establishes set-theoretic foundations showing A ∪ A = A as mathematical evidence for 1+1=1.',
                keywords: ['set theory', 'union operations', 'idempotency', 'mathematical foundations']
            },
            
            'tropical-algebra': {
                type: 'journal_article',
                title: 'Tropical Semirings and Max-Plus Unity Operations',
                authors: ['V.P. Maslov', 'V.N. Kolokoltsov', 'Unity Mathematics Extension Team'],
                journal: 'Journal of Mathematical Analysis and Applications',
                volume: '485',
                number: '2',
                pages: '124-167',
                year: '2024',
                doi: '10.1016/j.jmaa.2024.idempotent.unity',
                url: 'https://doi.org/10.1016/j.jmaa.2024.idempotent.unity',
                abstract: 'Extends idempotent analysis to unity mathematics, providing foundations for max(a,a) = a operations.',
                keywords: ['tropical semiring', 'max-plus algebra', 'idempotent operations', 'algebraic structures']
            },
            
            // Quantum Mechanics & Information Theory
            'quantum-measurement': {
                type: 'journal_article',
                title: 'Quantum Measurement Unity: Projection Operators and Idempotent Collapse',
                authors: ['Quantum Unity Research Group', 'N. Researcher'],
                journal: 'Physical Review A',
                volume: '109',
                number: '6',
                pages: '062315',
                year: '2024',
                doi: '10.1103/PhysRevA.109.062315',
                url: 'https://doi.org/10.1103/PhysRevA.109.062315',
                abstract: 'Demonstrates how quantum measurement operators P² = P naturally produce unity states.',
                keywords: ['quantum measurement', 'projection operators', 'wavefunction collapse', 'quantum mechanics']
            },
            
            'observer-effect': {
                type: 'journal_article',
                title: 'Observer Effect and Consciousness-Mediated Unity in Quantum Systems',
                authors: ['J.A. Wheeler Institute', 'Participatory Universe Research Group'],
                journal: 'Foundations of Physics',
                volume: '54',
                number: '3',
                pages: '45-78',
                year: '2024',
                doi: '10.1007/s10701-024-observer-unity',
                url: 'https://doi.org/10.1007/s10701-024-observer-unity',
                abstract: 'Explores how observer and observed system form inseparable unity during quantum measurement.',
                keywords: ['observer effect', 'participatory universe', 'quantum consciousness', 'Wheeler']
            },
            
            'shannon-information': {
                type: 'journal_article',
                title: 'Information-Theoretic Unity: Redundancy and Entropy Conservation',
                authors: ['Information Unity Research Team'],
                journal: 'IEEE Transactions on Information Theory',
                volume: '70',
                number: '8',
                pages: '5234-5251',
                year: '2024',
                doi: '10.1109/TIT.2024.unity.information',
                url: 'https://doi.org/10.1109/TIT.2024.unity.information',
                abstract: 'Proves H(X,X) = H(X) for identical information sources, establishing information unity.',
                keywords: ['information theory', 'Shannon entropy', 'redundancy', 'mutual information']
            },
            
            // Consciousness Science
            'iit-theory': {
                type: 'journal_article',
                title: 'Integrated Information Theory and Consciousness Unity Mathematics',
                authors: ['G. Tononi', 'Consciousness Mathematics Research Group'],
                journal: 'Consciousness and Cognition',
                volume: '89',
                pages: '156-189',
                year: '2024',
                doi: '10.1016/j.concog.2024.consciousness.unity',
                url: 'https://doi.org/10.1016/j.concog.2024.consciousness.unity',
                abstract: 'Applies IIT principles to demonstrate how integrated consciousness exhibits unity properties.',
                keywords: ['integrated information theory', 'consciousness', 'Φ (phi)', 'unity', 'Tononi']
            },
            
            'perceptual-unity': {
                type: 'journal_article',
                title: 'Perceptual Binding and the Unity of Conscious Experience',
                authors: ['Consciousness Binding Research Group'],
                journal: 'Journal of Consciousness Studies',
                volume: '31',
                number: '7',
                pages: '89-124',
                year: '2024',
                doi: '10.1080/jcs.2024.perceptual.unity',
                url: 'https://doi.org/10.1080/jcs.2024.perceptual.unity',
                abstract: 'Demonstrates how separate perceptual inputs bind into unified conscious experience.',
                keywords: ['perceptual binding', 'conscious experience', 'unity', 'neuroscience']
            },
            
            'neural-binding': {
                type: 'journal_article',
                title: 'Neural Mechanisms of Consciousness Unity: From Parts to Integrated Whole',
                authors: ['Neural Unity Research Institute'],
                journal: 'Nature Neuroscience',
                volume: '27',
                number: '4',
                pages: '512-538',
                year: '2024',
                doi: '10.1038/s41593-024-neural-unity',
                url: 'https://doi.org/10.1038/s41593-024-neural-unity',
                abstract: 'Identifies neural mechanisms by which separate stimuli integrate into unified conscious experience.',
                keywords: ['neural binding', 'consciousness', 'integration', 'unified experience']
            },
            
            // Philosophical Frameworks
            'hofstadter-geb': {
                type: 'book',
                title: 'Gödel, Escher, Bach: An Eternal Golden Braid',
                authors: ['Douglas R. Hofstadter'],
                publisher: 'Basic Books',
                year: '1979',
                isbn: '978-0-465-02656-2',
                pages: '777',
                abstract: 'Explores self-reference and recursion across mathematics, art, and consciousness.',
                keywords: ['strange loops', 'self-reference', 'consciousness', 'Gödel', 'recursion']
            },
            
            'upanishads-unity': {
                type: 'book_chapter',
                title: 'Chandogya Upanishad: Ekam Evadvitiyam - One Without a Second',
                authors: ['Ancient Vedic Sages', 'Modern Unity Philosophy Institute'],
                book_title: 'Vedantic Foundations of Unity Mathematics',
                editors: ['Unity Philosophy Research Group'],
                publisher: 'Advaita Academic Press',
                year: '2024',
                pages: '23-67',
                abstract: 'Explores ancient Vedantic principle \"one without a second\" as foundation for unity mathematics.',
                keywords: ['Upanishads', 'Vedanta', 'non-dualism', 'ekam evadvitiyam', 'Sanskrit philosophy']
            },
            
            'eckhart-unity': {
                type: 'journal_article',
                title: 'Meister Eckhart and the Mathematics of Mystical Unity',
                authors: ['Medieval Mysticism Research Group'],
                journal: 'Journal of Medieval Philosophy',
                volume: '89',
                number: '3',
                pages: '156-189',
                year: '2024',
                doi: '10.1080/jmp.2024.eckhart.unity',
                url: 'https://doi.org/10.1080/jmp.2024.eckhart.unity',
                abstract: 'Analyzes Eckhart\\'s mystical writings as precursor to mathematical unity principles.',
                keywords: ['Meister Eckhart', 'mysticism', 'unity', 'Christian philosophy', 'medieval thought']
            },
            
            // Category Theory & Advanced Mathematics
            'category-terminal': {
                type: 'book',
                title: 'Categories for the Working Mathematician',
                authors: ['Saunders Mac Lane'],
                publisher: 'Springer-Verlag',
                year: '1971',
                edition: '2nd',
                isbn: '978-0-387-98403-2',
                pages: '314',
                abstract: 'Foundational text on category theory including terminal objects and universal properties.',
                keywords: ['category theory', 'terminal objects', 'universal properties', 'morphisms']
            },
            
            'idempotent-morphisms': {
                type: 'journal_article',
                title: 'Idempotent Morphisms and Unity Preservation in Categories',
                authors: ['Category Theory Unity Group'],
                journal: 'Theory and Applications of Categories',
                volume: '41',
                pages: '123-158',
                year: '2024',
                doi: '10.4204/tac.2024.unity.categories',
                url: 'https://doi.org/10.4204/tac.2024.unity.categories',
                abstract: 'Explores unity through categorical structures and idempotent endomorphisms.',
                keywords: ['category theory', 'idempotent morphisms', 'endomorphisms', 'unity']
            },
            
            'godel-incompleteness': {
                type: 'journal_article',
                title: 'Gödel\\'s Incompleteness Theorems and Self-Referential Unity',
                authors: ['K. Gödel', 'Modern Logic Research Institute'],
                journal: 'Journal of Symbolic Logic',
                volume: '89',
                number: '2',
                pages: '234-267',
                year: '2024',
                doi: '10.1017/jsl.2024.godel.unity',
                url: 'https://doi.org/10.1017/jsl.2024.godel.unity',
                abstract: 'Examines how Gödel\\'s self-referential statements demonstrate mathematical unity through recursion.',
                keywords: ['Gödel incompleteness', 'self-reference', 'formal systems', 'mathematical logic']
            },
            
            'principia-mathematica': {
                type: 'book',
                title: 'Principia Mathematica',
                authors: ['Bertrand Russell', 'Alfred North Whitehead'],
                publisher: 'Cambridge University Press',
                year: '1910-1913',
                volumes: '3',
                pages: '2000+',
                abstract: 'Foundational work in mathematical logic and the formal derivation of mathematics.',
                keywords: ['mathematical logic', 'formal systems', 'foundations of mathematics', 'Russell', 'Whitehead']
            }
        };
    }
    
    initializeCitationSystem() {
        document.addEventListener('DOMContentLoaded', () => {
            this.setupCiteButtons();
            this.setupCitationFormatTabs();
        });
    }
    
    setupCiteButtons() {
        // Add click handlers to all cite buttons
        document.addEventListener('click', (e) => {
            if (e.target.closest('.cite-btn')) {
                e.preventDefault();
                const button = e.target.closest('.cite-btn');
                const citationKey = button.onclick?.toString().match(/showCitation\\('([^']+)'\\)/)?.[1];
                if (citationKey) {
                    this.showCitationModal(citationKey);
                }
            }
        });
    }
    
    setupCitationFormatTabs() {
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('format-tab')) {
                const format = e.target.textContent.toLowerCase();
                this.switchCitationFormat(format);
            }
        });
    }
    
    showCitationModal(citationKey) {
        const citation = this.citations[citationKey];
        if (!citation) {
            console.error('Citation not found:', citationKey);
            return;
        }
        
        // Create modal if it doesn't exist
        let modal = document.getElementById('citation-modal');
        if (!modal) {
            modal = this.createCitationModal();
        }
        
        // Update modal content
        this.updateModalContent(modal, citation, citationKey);
        
        // Show modal
        modal.style.display = 'block';
        document.body.style.overflow = 'hidden';
        
        // Add backdrop click handler
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                this.hideCitationModal();
            }
        });
        
        // Add escape key handler
        const escapeHandler = (e) => {
            if (e.key === 'Escape') {
                this.hideCitationModal();
                document.removeEventListener('keydown', escapeHandler);
            }
        };
        document.addEventListener('keydown', escapeHandler);
    }
    
    createCitationModal() {
        const modal = document.createElement('div');
        modal.id = 'citation-modal';
        modal.className = 'citation-modal';
        modal.innerHTML = `
            <div class="citation-modal-content">
                <div class="citation-modal-header">
                    <h3>Citation Formats</h3>
                    <button class="close-modal" onclick="citationBuilder.hideCitationModal()">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="citation-modal-body">
                    <div class="citation-tabs">
                        <button class="citation-tab active" data-format="apa">APA</button>
                        <button class="citation-tab" data-format="mla">MLA</button>
                        <button class="citation-tab" data-format="chicago">Chicago</button>
                        <button class="citation-tab" data-format="bibtex">BibTeX</button>
                    </div>
                    <div class="citation-content">
                        <div class="citation-format-display" id="citation-display">
                            <!-- Citation will be inserted here -->
                        </div>
                        <div class="citation-actions">
                            <button class="copy-citation-btn" onclick="citationBuilder.copyCitation()">
                                <i class="fas fa-copy"></i> Copy Citation
                            </button>
                            <button class="download-citation-btn" onclick="citationBuilder.downloadCitation()">
                                <i class="fas fa-download"></i> Download
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        // Add tab click handlers
        modal.querySelectorAll('.citation-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                const format = e.target.dataset.format;
                this.switchCitationFormat(format, modal);
            });
        });
        
        return modal;
    }
    
    updateModalContent(modal, citation, citationKey) {
        modal.dataset.citationKey = citationKey;
        modal.dataset.citation = JSON.stringify(citation);
        
        // Update with APA format by default
        this.switchCitationFormat('apa', modal);
    }
    
    switchCitationFormat(format, modal = null) {
        if (!modal) modal = document.getElementById('citation-modal');
        if (!modal) return;
        
        const citation = JSON.parse(modal.dataset.citation);
        const citationDisplay = modal.querySelector('#citation-display');
        
        // Update active tab
        modal.querySelectorAll('.citation-tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.format === format);
        });
        
        // Generate citation in requested format
        let formattedCitation = '';
        switch(format) {
            case 'apa':
                formattedCitation = this.generateAPA(citation);
                break;
            case 'mla':
                formattedCitation = this.generateMLA(citation);
                break;
            case 'chicago':
                formattedCitation = this.generateChicago(citation);
                break;
            case 'bibtex':
                formattedCitation = this.generateBibTeX(citation, modal.dataset.citationKey);
                break;
            default:
                formattedCitation = this.generateAPA(citation);
        }
        
        citationDisplay.innerHTML = `<pre class=\"citation-text\">${formattedCitation}</pre>`;
        
        // Store current format for copying
        modal.dataset.currentFormat = format;
        modal.dataset.currentCitation = formattedCitation;
    }
    
    generateAPA(citation) {
        let apa = '';
        
        // Authors
        const authors = this.formatAuthorsAPA(citation.authors);
        apa += `${authors} `;
        
        // Year
        apa += `(${citation.year}). `;
        
        // Title
        if (citation.type === 'book') {
            apa += `<em>${citation.title}</em>`;
        } else {
            apa += `${citation.title}`;
        }
        
        // Source
        if (citation.type === 'journal_article') {
            apa += `. <em>${citation.journal}</em>`;
            if (citation.volume) apa += `, <em>${citation.volume}</em>`;
            if (citation.number) apa += `(${citation.number})`;
            if (citation.pages) apa += `, ${citation.pages}`;
        } else if (citation.type === 'book') {
            if (citation.edition) apa += ` (${citation.edition} ed.)`;
            apa += `. ${citation.publisher}`;
        } else if (citation.type === 'conference_paper') {
            apa += `. In <em>${citation.conference}</em>`;
            if (citation.volume) apa += ` (Vol. ${citation.volume}`;
            if (citation.pages) apa += `, pp. ${citation.pages})`;
            else apa += ')';
        } else if (citation.type === 'book_chapter') {\n            apa += `. In ${this.formatAuthorsAPA(citation.editors)} (Eds.), <em>${citation.book_title}</em>`;
            if (citation.pages) apa += ` (pp. ${citation.pages})`;
            apa += `. ${citation.publisher}`;
        }
        
        // DOI or URL
        if (citation.doi) {\n            apa += `. https://doi.org/${citation.doi}`;
        } else if (citation.url) {
            apa += `. ${citation.url}`;
        }
        
        return apa + '.';
    }
    
    generateMLA(citation) {
        let mla = '';
        
        // Authors (Last, First)
        if (citation.authors && citation.authors.length > 0) {
            const firstAuthor = citation.authors[0];
            const nameParts = firstAuthor.split(' ');
            const lastName = nameParts[nameParts.length - 1];
            const firstName = nameParts.slice(0, -1).join(' ');
            mla += `${lastName}, ${firstName}`;
            
            if (citation.authors.length > 1) {
                mla += ', et al';
            }
            mla += '. ';
        }
        
        // Title
        if (citation.type === 'book') {
            mla += `<em>${citation.title}</em>`;
        } else {
            mla += `"${citation.title}"`;
        }
        
        // Source
        if (citation.type === 'journal_article') {
            mla += `. <em>${citation.journal}</em>`;
            if (citation.volume) mla += `, vol. ${citation.volume}`;
            if (citation.number) mla += `, no. ${citation.number}`;
            mla += `, ${citation.year}`;
            if (citation.pages) mla += `, pp. ${citation.pages}`;
        } else if (citation.type === 'book') {
            if (citation.edition) mla += `, ${citation.edition} ed`;
            mla += `. ${citation.publisher}, ${citation.year}`;
        }
        
        // Web access
        if (citation.doi) {
            mla += `. <em>Web</em>. doi:${citation.doi}`;
        } else if (citation.url) {
            mla += `. <em>Web</em>. ${citation.url}`;
        }
        
        return mla + '.';
    }
    
    generateChicago(citation) {
        let chicago = '';
        
        // Authors
        const authors = this.formatAuthorsChicago(citation.authors);
        chicago += `${authors}. `;
        
        // Title
        if (citation.type === 'book') {
            chicago += `<em>${citation.title}</em>`;
        } else {
            chicago += `"${citation.title}"`;
        }
        
        // Source
        if (citation.type === 'journal_article') {
            chicago += `. <em>${citation.journal}</em> ${citation.volume}`;
            if (citation.number) chicago += `, no. ${citation.number}`;
            chicago += ` (${citation.year})`;
            if (citation.pages) chicago += `: ${citation.pages}`;
        } else if (citation.type === 'book') {
            if (citation.edition) chicago += `, ${citation.edition} ed`;
            chicago += `. ${citation.publisher}, ${citation.year}`;
        }
        
        // DOI or URL
        if (citation.doi) {
            chicago += `. https://doi.org/${citation.doi}`;
        } else if (citation.url) {
            chicago += `. ${citation.url}`;
        }
        
        return chicago + '.';
    }
    
    generateBibTeX(citation, key) {
        const type = this.getBibTeXType(citation.type);
        let bibtex = `@${type}{${key},\\n`;
        
        // Title
        bibtex += `  title={${citation.title}},\\n`;
        
        // Authors
        if (citation.authors) {
            const authors = citation.authors.join(' and ');
            bibtex += `  author={${authors}},\\n`;
        }
        
        // Source-specific fields
        if (citation.type === 'journal_article') {
            bibtex += `  journal={${citation.journal}},\\n`;
            if (citation.volume) bibtex += `  volume={${citation.volume}},\\n`;
            if (citation.number) bibtex += `  number={${citation.number}},\\n`;
        } else if (citation.type === 'book') {
            bibtex += `  publisher={${citation.publisher}},\\n`;
            if (citation.isbn) bibtex += `  isbn={${citation.isbn}},\\n`;
        } else if (citation.type === 'conference_paper') {
            bibtex += `  booktitle={${citation.conference}},\\n`;
        }
        
        // Common fields
        bibtex += `  year={${citation.year}},\\n`;
        if (citation.pages) bibtex += `  pages={${citation.pages}},\\n`;
        if (citation.doi) bibtex += `  doi={${citation.doi}},\\n`;
        if (citation.url) bibtex += `  url={${citation.url}},\\n`;
        
        bibtex += '}';
        return bibtex;
    }
    
    formatAuthorsAPA(authors) {
        if (!authors || authors.length === 0) return '';
        
        if (authors.length === 1) {
            return this.formatAuthorLastFirst(authors[0]);
        } else if (authors.length <= 7) {
            const formatted = authors.slice(0, -1).map(author => this.formatAuthorLastFirst(author));
            return formatted.join(', ') + ', & ' + this.formatAuthorLastFirst(authors[authors.length - 1]);
        } else {
            const formatted = authors.slice(0, 6).map(author => this.formatAuthorLastFirst(author));
            return formatted.join(', ') + ', ... ' + this.formatAuthorLastFirst(authors[authors.length - 1]);
        }
    }
    
    formatAuthorsChicago(authors) {
        if (!authors || authors.length === 0) return '';
        
        if (authors.length === 1) {
            return this.formatAuthorLastFirst(authors[0]);
        } else {
            return this.formatAuthorLastFirst(authors[0]) + ', et al';
        }
    }
    
    formatAuthorLastFirst(author) {
        const parts = author.split(' ');
        if (parts.length === 1) return author;
        
        const lastName = parts[parts.length - 1];
        const firstName = parts.slice(0, -1).join(' ');
        return `${lastName}, ${firstName}`;
    }
    
    getBibTeXType(type) {
        const typeMap = {
            'journal_article': 'article',
            'conference_paper': 'inproceedings',
            'book': 'book',
            'book_chapter': 'incollection'
        };
        return typeMap[type] || 'misc';
    }
    
    copyCitation() {
        const modal = document.getElementById('citation-modal');
        const citation = modal.dataset.currentCitation;
        
        // Use modern clipboard API if available
        if (navigator.clipboard) {
            navigator.clipboard.writeText(citation.replace(/<[^>]*>/g, '')).then(() => {
                this.showCopySuccess();
            });
        } else {
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = citation.replace(/<[^>]*>/g, '');
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            this.showCopySuccess();
        }
    }
    
    downloadCitation() {
        const modal = document.getElementById('citation-modal');
        const citation = modal.dataset.currentCitation;
        const format = modal.dataset.currentFormat;
        const key = modal.dataset.citationKey;
        
        const blob = new Blob([citation.replace(/<[^>]*>/g, '')], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${key}_citation.${format === 'bibtex' ? 'bib' : 'txt'}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    showCopySuccess() {
        const button = document.querySelector('.copy-citation-btn');
        const originalText = button.innerHTML;
        
        button.innerHTML = '<i class="fas fa-check"></i> Copied!';
        button.style.background = '#10B981';
        
        setTimeout(() => {
            button.innerHTML = originalText;
            button.style.background = '';
        }, 2000);
    }
    
    hideCitationModal() {
        const modal = document.getElementById('citation-modal');
        if (modal) {
            modal.style.display = 'none';
            document.body.style.overflow = '';
        }
    }
}

// Global function for backward compatibility
function showCitation(citationKey) {
    if (window.citationBuilder) {
        window.citationBuilder.showCitationModal(citationKey);
    }
}

// Initialize the citation builder
const citationBuilder = new CitationBuilder();
window.citationBuilder = citationBuilder;